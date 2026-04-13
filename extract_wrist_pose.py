#!/usr/bin/env python3
"""
Extract 3D wrist poses from ego camera (07) using MediaPipe + depth.
Camera frame is used directly (no coordinate transform needed).

Supports either:
1. A single sequence directory containing camera_params.json and 07/RGB, or
2. A parent directory whose immediate children are multiple sequence directories.
"""
import argparse, json
import numpy as np
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEPTH_SCALE = 0.001  # Orbbec: uint16 mm -> meters

def sort_key(path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

def get_by_alias(mapping, aliases):
    if not isinstance(mapping, dict):
        return None
    normalized = {str(k).lower(): v for k, v in mapping.items()}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None

def normalize_intrinsic(node):
    if node is None:
        return None
    if not isinstance(node, dict):
        return None
    intrinsic = get_by_alias(node, ["intrinsic", "intrinsics"])
    if intrinsic is None:
        intrinsic = node
    if not isinstance(intrinsic, dict):
        return None
    result = {}
    for key in ["fx", "fy", "cx", "cy", "width", "height"]:
        value = get_by_alias(intrinsic, [key])
        if value is not None:
            result[key] = value
    return result if all(k in result for k in ["fx", "fy", "cx", "cy"]) else None

def load_cam_params(data_dir, cam_id):
    with open(Path(data_dir) / "camera_params.json") as f:
        p = json.load(f)
    cam = get_by_alias(p, [cam_id, str(int(cam_id)) if str(cam_id).isdigit() else cam_id])
    if cam is None:
        raise KeyError(f"Camera {cam_id} not found in camera_params.json")
    rgb_intr = (
        normalize_intrinsic(get_by_alias(cam, ["RGB", "rgb", "color", "Color"]))
        or normalize_intrinsic(get_by_alias(cam, ["rgb_intrinsic", "color_intrinsic"]))
        or normalize_intrinsic(cam)
    )
    depth_intr = (
        normalize_intrinsic(get_by_alias(cam, ["Depth", "depth"]))
        or normalize_intrinsic(get_by_alias(cam, ["depth_intrinsic"]))
        or rgb_intr
    )
    if rgb_intr is None or depth_intr is None:
        raise KeyError(
            f"Could not parse RGB/Depth intrinsics for camera {cam_id} from camera_params.json"
        )
    return {
        "rgb": rgb_intr,
        "depth": depth_intr,
    }

def px_to_3d(u, v, d_raw, intr):
    z = d_raw * DEPTH_SCALE
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return np.array([x, y, z], dtype=np.float32)

def get_depth_at(depth_img, u, v, H, W, search_r=4):
    """Get valid depth near pixel, searching outward if needed."""
    for r in range(search_r + 1):
        for du in range(-r, r+1):
            for dv in range(-r, r+1):
                if abs(du) != r and abs(dv) != r:
                    continue
                nu = int(np.clip(u + du, 0, W-1))
                nv = int(np.clip(v + dv, 0, H-1))
                d = depth_img[nv, nu]
                if 100 < d < 3000:
                    return d
    return 0

def landmark_to_3d(lms, idx, depth_img, intr, H, W, fallback_depth=0):
    """Lift a landmark to 3D, falling back to wrist depth if local depth is missing."""
    lm = lms[idx]
    u = int(np.clip(lm.x * W, 0, W-1))
    v = int(np.clip(lm.y * H, 0, H-1))
    d = get_depth_at(depth_img, u, v, H, W)
    if d == 0:
        d = fallback_depth
    if d == 0:
        return None
    return px_to_3d(u, v, d, intr)

def compute_orientation(lms, depth_img, intr, H, W):
    """Wrist orientation from palm plane (lm 0, 5, 17)."""
    lm0 = lms[0]
    u0 = int(np.clip(lm0.x * W, 0, W-1))
    v0 = int(np.clip(lm0.y * H, 0, H-1))
    wrist_depth = get_depth_at(depth_img, u0, v0, H, W)
    p0 = landmark_to_3d(lms, 0, depth_img, intr, H, W, wrist_depth)
    p5 = landmark_to_3d(lms, 5, depth_img, intr, H, W, wrist_depth)
    p17 = landmark_to_3d(lms, 17, depth_img, intr, H, W, wrist_depth)
    if p0 is None or p5 is None or p17 is None:
        return np.eye(3)
    v1 = p5 - p0
    v2 = p17 - p0
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.eye(3)
    y_axis = v1 / n1
    normal = np.cross(y_axis, v2)
    nn = np.linalg.norm(normal)
    if nn < 1e-6:
        return np.eye(3)
    z_axis = normal / nn

    # Build a right-handed orthonormal frame:
    # x = y × z so that x × y = z.
    x_axis = np.cross(y_axis, z_axis)
    nx = np.linalg.norm(x_axis)
    if nx < 1e-6:
        return np.eye(3)
    x_axis /= nx
    y_axis = np.cross(z_axis, x_axis)
    ny = np.linalg.norm(y_axis)
    if ny < 1e-6:
        return np.eye(3)
    y_axis /= ny

    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1.0
    return rot

def compute_gripper_distance(lms, depth_img, intr, H, W, wrist_depth):
    """Use thumb/index fingertip distance as a proxy for gripper opening."""
    p4 = landmark_to_3d(lms, 4, depth_img, intr, H, W, wrist_depth)
    p8 = landmark_to_3d(lms, 8, depth_img, intr, H, W, wrist_depth)
    if p4 is None or p8 is None:
        return np.float32(0.0)
    return np.float32(np.linalg.norm(p4 - p8))

def interpolate_invalid(poses, valid):
    result = poses.copy()
    vi = np.where(valid)[0]
    if len(vi) < 2:
        return result
    ii = np.where(~valid)[0]
    for d in range(poses.shape[1]):
        result[ii, d] = np.interp(ii, vi, poses[vi, d])
    return result

def smooth(poses, method):
    T = len(poses)
    win = min(11, T if T % 2 == 1 else T - 1)
    if win < 5 or method == "none":
        return poses
    s = poses.copy()
    for d in range(poses.shape[1]):
        if method == "savgol":
            s[:, d] = savgol_filter(poses[:, d], win, 3)
        elif method == "ema":
            for t in range(1, T):
                s[t, d] = 0.3 * poses[t, d] + 0.7 * s[t-1, d]
        elif method == "median_then_savgol":
            s[:, d] = median_filter(poses[:, d], size=5)
            s[:, d] = savgol_filter(s[:, d], win, 3)
    return s

def is_sequence_dir(path, cam_id):
    return (path / "camera_params.json").is_file() and (path / cam_id / "RGB").is_dir()

def discover_sequence_dirs(root_dir, cam_id):
    root_dir = Path(root_dir)
    if is_sequence_dir(root_dir, cam_id):
        return [root_dir]
    seq_dirs = sorted(
        [path for path in root_dir.iterdir() if path.is_dir() and is_sequence_dir(path, cam_id)],
        key=sort_key,
    )
    if not seq_dirs:
        raise FileNotFoundError(
            f"No sequence directories found under {root_dir}. "
            f"Expected either {root_dir / cam_id / 'RGB'} or child folders like */{cam_id}/RGB."
        )
    return seq_dirs

def resolve_output_path(output_arg, seq_dir, batch_mode):
    if not output_arg:
        return seq_dir / "wrist_poses.npz"
    output_path = Path(output_arg)
    if not batch_mode and output_path.suffix == ".npz":
        return output_path
    if batch_mode and output_path.suffix == ".npz":
        batch_dir = output_path.parent if str(output_path.parent) not in ("", ".") else Path.cwd()
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir / f"{seq_dir.name}_{output_path.name}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{seq_dir.name}_wrist_poses.npz"

def process_sequence(seq_dir, output_path, detector, cam_id, smooth_method):
    params = load_cam_params(seq_dir, cam_id)
    rgb_intr = params["rgb"]
    depth_intr = params["depth"]

    rgb_files = sorted((seq_dir / cam_id / "RGB").glob("*.jpg"))
    depth_dir = seq_dir / cam_id / "Depth"
    T = len(rgb_files)
    if T == 0:
        raise FileNotFoundError(f"No RGB frames found in {seq_dir / cam_id / 'RGB'}")
    sample_img = cv2.imread(str(rgb_files[0]))
    if sample_img is None:
        raise FileNotFoundError(f"Failed to read sample RGB frame: {rgb_files[0]}")
    H, W = sample_img.shape[:2]
    rgb_intr.setdefault("height", H)
    rgb_intr.setdefault("width", W)
    depth_intr.setdefault("height", H)
    depth_intr.setdefault("width", W)
    print(f"Processing sequence {seq_dir.name}: {T} frames from camera {cam_id}")

    left_poses  = np.zeros((T, 6), dtype=np.float32)
    right_poses = np.zeros((T, 6), dtype=np.float32)
    left_gripper  = np.zeros(T, dtype=np.float32)
    right_gripper = np.zeros(T, dtype=np.float32)
    left_valid  = np.zeros(T, dtype=bool)
    right_valid = np.zeros(T, dtype=bool)

    for i, rgb_path in enumerate(rgb_files):
        if i % 50 == 0:
            print(f"  frame {i}/{T}  left_det={left_valid[:i].sum()}  right_det={right_valid[:i].sum()}")

        img_bgr = cv2.imread(str(rgb_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        depth_path = depth_dir / (rgb_path.stem + ".png")
        depth_img  = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            depth_img = np.zeros((H, W), dtype=np.uint16)

        result = detector.detect(mp_img)

        for lms, handedness in zip(result.hand_landmarks, result.handedness):
            side = handedness[0].category_name.lower()
            side = "right" if side == "left" else "left"

            lm0 = lms[0]
            u = int(np.clip(lm0.x * W, 0, W-1))
            v = int(np.clip(lm0.y * H, 0, H-1))
            d = get_depth_at(depth_img, u, v, H, W)
            if d == 0:
                continue

            wrist_pos = px_to_3d(u, v, d, depth_intr)
            rot_mat   = compute_orientation(lms, depth_img, depth_intr, H, W)
            euler     = R.from_matrix(rot_mat).as_euler("xyz")
            pose      = np.concatenate([wrist_pos, euler])
            gripper   = compute_gripper_distance(lms, depth_img, depth_intr, H, W, d)

            if side == "left":
                left_poses[i]  = pose
                left_gripper[i] = gripper
                left_valid[i]  = True
            else:
                right_poses[i] = pose
                right_gripper[i] = gripper
                right_valid[i] = True

    print(f"Detection: left={left_valid.sum()}/{T} ({100*left_valid.mean():.1f}%)  right={right_valid.sum()}/{T} ({100*right_valid.mean():.1f}%)")

    left_poses  = interpolate_invalid(left_poses,  left_valid)
    right_poses = interpolate_invalid(right_poses, right_valid)
    left_poses  = smooth(left_poses,  smooth_method)
    right_poses = smooth(right_poses, smooth_method)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path),
             left_wrist_poses=left_poses,
             right_wrist_poses=right_poses,
             left_gripper=left_gripper,
             right_gripper=right_gripper,
             left_valid=left_valid,
             right_valid=right_valid,
             fps=np.float32(30.0))
    print(f"Saved -> {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Single sequence dir or a parent dir containing multiple sequence subdirectories")
    ap.add_argument("--output", default=None,
                    help="For single sequence: output .npz path. For batch mode: output directory. "
                         "If omitted, saves wrist_poses.npz inside each sequence directory.")
    ap.add_argument("--model_path", default="/home/ubuntu/WorkSpace/ZYC/hamer/_DATA/mediapipe/hand_landmarker.task")
    ap.add_argument("--ego_cam_id", default="07", help="Camera folder id used as the ego view for hand pose extraction")
    ap.add_argument("--cam_id", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--smooth_method", default="median_then_savgol",
                    choices=["savgol","ema","median_then_savgol","none"])
    args = ap.parse_args()

    ego_cam_id = args.ego_cam_id or args.cam_id or "07"
    data_dir = Path(args.data_dir)
    seq_dirs = discover_sequence_dirs(data_dir, ego_cam_id)
    batch_mode = len(seq_dirs) > 1

    base_opts = mp_python.BaseOptions(model_asset_path=args.model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    print(f"Discovered {len(seq_dirs)} sequence(s) under {data_dir}")
    print(f"Ego camera mapping: cam_high->{ego_cam_id}")
    for seq_dir in seq_dirs:
        out = resolve_output_path(args.output, seq_dir, batch_mode)
        process_sequence(seq_dir, out, detector, ego_cam_id, args.smooth_method)

if __name__ == "__main__":
    main()
