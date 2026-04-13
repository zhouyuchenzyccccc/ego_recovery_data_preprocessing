#!/usr/bin/env python3
"""
Visualize extracted wrist poses on ego camera frames.
Draws MediaPipe landmarks, 3D coordinate axes, and pose text.

Usage:
    python3 visualize_wrist_pose.py         --data_dir /home/ubuntu/orbbec/src/sync/test/test/ego2         --poses /home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/wrist_poses.npz         --output /home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/wrist_viz.mp4

"""
import argparse, json
import numpy as np
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEPTH_SCALE = 0.001

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def get_by_alias(mapping, aliases):
    if not isinstance(mapping, dict):
        return None
    normalized = {str(k).lower(): v for k, v in mapping.items()}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None

def normalize_intrinsic(node):
    if node is None or not isinstance(node, dict):
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

def proj(pt3d, intr):
    x, y, z = pt3d
    if z < 1e-4:
        return None
    u = int(x / z * intr["fx"] + intr["cx"])
    v = int(y / z * intr["fy"] + intr["cy"])
    return (u, v)

def draw_axes(img, pos, euler, intr, length=0.05):
    rot = R.from_euler("xyz", euler).as_matrix()
    o = proj(pos, intr)
    if o is None:
        return
    for i, color in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
        ep = proj(pos + rot[:, i] * length, intr)
        if ep is not None:
            cv2.arrowedLine(img, o, ep, color, 2, tipLength=0.3)

def draw_hand(img, lms, H, W, color):
    pts = [(int(np.clip(lm.x*W,0,W-1)), int(np.clip(lm.y*H,0,H-1))) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 1)
    for pt in pts:
        cv2.circle(img, pt, 3, color, -1)
    cv2.circle(img, pts[0], 7, (0,0,255), 2)  # wrist highlight

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="/home/ubuntu/orbbec/src/sync/test/test/ego2")
    ap.add_argument("--poses",      default="/home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/wrist_poses.npz")
    ap.add_argument("--model_path", default="/home/ubuntu/WorkSpace/ZYC/hamer/_DATA/mediapipe/hand_landmarker.task")
    ap.add_argument("--output",     default="/home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/wrist_viz.mp4")
    ap.add_argument("--ego_cam_id", default="07", help="Camera folder id used as the ego view")
    ap.add_argument("--cam_id",     default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fps",        type=float, default=30.0)
    ap.add_argument("--max_frames", type=int,   default=None)
    args = ap.parse_args()

    ego_cam_id = args.ego_cam_id or args.cam_id or "07"
    data_dir = Path(args.data_dir)
    params   = load_cam_params(data_dir, ego_cam_id)
    rgb_intr  = params["rgb"]
    depth_intr = params["depth"]
    rgb_files = sorted((data_dir / ego_cam_id / "RGB").glob("*.jpg"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB frames found in {data_dir / ego_cam_id / 'RGB'}")
    sample_img = cv2.imread(str(rgb_files[0]))
    if sample_img is None:
        raise FileNotFoundError(f"Failed to read sample RGB frame: {rgb_files[0]}")
    H, W = sample_img.shape[:2]
    rgb_intr.setdefault("height", H)
    rgb_intr.setdefault("width", W)
    depth_intr.setdefault("height", H)
    depth_intr.setdefault("width", W)

    data = np.load(args.poses)
    left_poses  = data["left_wrist_poses"]
    right_poses = data["right_wrist_poses"]
    left_valid  = data["left_valid"]
    right_valid = data["right_valid"]

    if args.max_frames:
        rgb_files = rgb_files[:args.max_frames]
    T = len(rgb_files)

    base_opts = mp_python.BaseOptions(model_asset_path=args.model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    for i, rgb_path in enumerate(rgb_files):
        if i % 50 == 0:
            print(f"  frame {i}/{T}")

        img = cv2.imread(str(rgb_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result  = detector.detect(mp_img)

        # Draw live landmarks
        for lms, handedness in zip(result.hand_landmarks, result.handedness):
            side  = handedness[0].category_name.lower()
            side  = "right" if side == "left" else "left"
            color = (0,255,255) if side == "left" else (255,165,0)
            draw_hand(img, lms, H, W, color)

        # Draw smoothed pose axes + text
        for side, poses, valid in [("left", left_poses, left_valid), ("right", right_poses, right_valid)]:
            if i >= len(poses):
                continue
            pos   = poses[i, :3]
            euler = poses[i, 3:6]
            color = (0,255,255) if side == "left" else (255,165,0)
            draw_axes(img, pos, euler, depth_intr, length=0.05)

            o = proj(pos, depth_intr)
            if o is not None:
                is_valid = bool(valid[i]) if i < len(valid) else False
                label = f"{side[0].upper()} xyz=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})m"
                status = "DET" if is_valid else "INTERP"
                s_color = (0,220,0) if is_valid else (0,140,255)
                tx, ty = min(o[0]+12, W-200), o[1]
                cv2.putText(img, label,  (tx, ty),    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color,   1, cv2.LINE_AA)
                cv2.putText(img, status, (tx, ty+16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, s_color, 1, cv2.LINE_AA)

        cv2.putText(img, f"Frame {i:04d}/{T}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, "Axes: X=red Y=green Z=blue", (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
