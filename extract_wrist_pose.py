#!/usr/bin/env python3
"""
Extract 3D wrist poses from ego camera (07) using MediaPipe + depth.
Camera frame is used directly (no coordinate transform needed).

Usage:
    python3 extract_wrist_pose.py --data_dir /home/ubuntu/orbbec/src/sync/test/test/ego4
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

def load_cam_params(data_dir, cam_id):
    with open(Path(data_dir) / "camera_params.json") as f:
        p = json.load(f)
    cam = p[cam_id]
    return {
        "rgb": cam["RGB"]["intrinsic"],
        "depth": cam["Depth"]["intrinsic"],
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

def compute_orientation(lms, depth_img, intr, H, W):
    """Wrist orientation from palm plane (lm 0, 5, 17)."""
    def lm3d(idx):
        lm = lms[idx]
        u = int(np.clip(lm.x * W, 0, W-1))
        v = int(np.clip(lm.y * H, 0, H-1))
        d = get_depth_at(depth_img, u, v, H, W)
        if d == 0:
            lm0 = lms[0]
            u0 = int(np.clip(lm0.x * W, 0, W-1))
            v0 = int(np.clip(lm0.y * H, 0, H-1))
            d = get_depth_at(depth_img, u0, v0, H, W)
        return px_to_3d(u, v, d, intr)

    p0, p5, p17 = lm3d(0), lm3d(5), lm3d(17)
    v1 = p5 - p0
    v2 = p17 - p0
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.eye(3)
    v1 /= n1
    normal = np.cross(v1, v2)
    nn = np.linalg.norm(normal)
    if nn < 1e-6:
        return np.eye(3)
    normal /= nn
    v2c = np.cross(normal, v1)
    return np.stack([v2c, v1, normal], axis=1)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/ubuntu/orbbec/src/sync/test/test/ego2")
    ap.add_argument("--output", default="/home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/wrist_poses.npz")
    ap.add_argument("--model_path", default="/home/ubuntu/WorkSpace/ZYC/hamer/_DATA/mediapipe/hand_landmarker.task")
    ap.add_argument("--cam_id", default="07")
    ap.add_argument("--smooth_method", default="median_then_savgol",
                    choices=["savgol","ema","median_then_savgol","none"])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    params = load_cam_params(data_dir, args.cam_id)
    rgb_intr = params["rgb"]
    depth_intr = params["depth"]
    H, W = int(rgb_intr["height"]), int(rgb_intr["width"])

    rgb_files = sorted((data_dir / args.cam_id / "RGB").glob("*.jpg"))
    depth_dir = data_dir / args.cam_id / "Depth"
    T = len(rgb_files)
    print(f"Processing {T} frames from camera {args.cam_id}")

    base_opts = mp_python.BaseOptions(model_asset_path=args.model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    left_poses  = np.zeros((T, 6), dtype=np.float32)
    right_poses = np.zeros((T, 6), dtype=np.float32)
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
            # MediaPipe assumes selfie/front-facing -> flip for fixed ego cam
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

            if side == "left":
                left_poses[i]  = pose
                left_valid[i]  = True
            else:
                right_poses[i] = pose
                right_valid[i] = True

    print(f"Detection: left={left_valid.sum()}/{T} ({100*left_valid.mean():.1f}%)  right={right_valid.sum()}/{T} ({100*right_valid.mean():.1f}%)")

    left_poses  = interpolate_invalid(left_poses,  left_valid)
    right_poses = interpolate_invalid(right_poses, right_valid)
    left_poses  = smooth(left_poses,  args.smooth_method)
    right_poses = smooth(right_poses, args.smooth_method)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out),
             left_wrist_poses=left_poses,
             right_wrist_poses=right_poses,
             left_valid=left_valid,
             right_valid=right_valid,
             fps=np.float32(30.0))
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
