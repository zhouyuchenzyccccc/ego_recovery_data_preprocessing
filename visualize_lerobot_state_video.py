#!/usr/bin/env python3
"""
Overlay LeRobot 14-D state/action data directly on episode videos.

This script is intended for the human LeRobot dataset produced by
`convert_to_lerobot.py`, where:
  - observation.state is a 14-D absolute wrist pose + gripper vector
  - action is also a 14-D absolute target pose vector

The script draws, for both hands:
  1. current state pose axes,
  2. target action pose axes,
  3. current->target position link,
  4. state/action text near the wrist.

Because these 3D poses are stored in the ego-camera frame, this script is
designed to overlay on `observation.images.cam_high` only.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


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


def load_info(dataset_dir):
    info_path = Path(dataset_dir) / "meta" / "info.json"
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def locate_episode_paths(dataset_dir, info, episode_index, video_key):
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunk_size
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    video_path = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    parquet_path = Path(dataset_dir) / data_path.format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    video_path = Path(dataset_dir) / video_path.format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    return parquet_path, video_path


def load_intrinsics(source_seq_dir, ego_cam_id):
    with open(Path(source_seq_dir) / "camera_params.json", encoding="utf-8") as f:
        params = json.load(f)
    aliases = [str(ego_cam_id)]
    if str(ego_cam_id).isdigit():
        aliases.append(str(int(ego_cam_id)))
    cam = get_by_alias(params, aliases)
    if cam is None:
        raise KeyError(f"Camera {ego_cam_id} not found in camera_params.json")
    intr = (
        normalize_intrinsic(get_by_alias(cam, ["RGB", "rgb", "color", "Color"]))
        or normalize_intrinsic(get_by_alias(cam, ["rgb_intrinsic", "color_intrinsic"]))
        or normalize_intrinsic(cam)
    )
    if intr is None:
        raise KeyError(f"Could not parse RGB intrinsics for camera {ego_cam_id}")
    return intr


def load_episode_arrays(parquet_path, info):
    try:
        table = pq.ParquetFile(parquet_path).read()
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read parquet file {parquet_path}. "
            "This is usually a pyarrow/parquet compatibility issue."
        ) from exc

    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    if len(state_names) != 14 or len(action_names) != 14:
        raise ValueError(
            "This script expects 14-D state/action data. "
            f"Got state={len(state_names)}, action={len(action_names)}."
        )
    return states, actions


def proj(pt3d, intr):
    x, y, z = pt3d
    if z < 1e-5:
        return None
    u = int(x / z * intr["fx"] + intr["cx"])
    v = int(y / z * intr["fy"] + intr["cy"])
    return (u, v)


def draw_axes(img, pos, euler, intr, length=0.04, thickness=2, alpha=1.0):
    overlay = img.copy()
    rot = R.from_euler("xyz", euler).as_matrix()
    origin = proj(pos, intr)
    if origin is None:
        return
    for axis_idx, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        endpoint = proj(pos + rot[:, axis_idx] * length, intr)
        if endpoint is not None:
            cv2.arrowedLine(overlay, origin, endpoint, color, thickness, tipLength=0.28)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    else:
        img[:] = overlay


def hand_slices(side):
    if side == "left":
        return slice(0, 6), 6
    return slice(7, 13), 13


def draw_hand_overlay(frame, side, state_row, action_row, intr):
    pose_slice, grip_idx = hand_slices(side)
    state_pose = state_row[pose_slice]
    action_pose = action_row[pose_slice]
    state_grip = float(state_row[grip_idx])
    action_grip = float(action_row[grip_idx])

    state_uv = proj(state_pose[:3], intr)
    action_uv = proj(action_pose[:3], intr)
    if state_uv is None:
        return

    if action_uv is not None:
        cv2.line(frame, state_uv, action_uv, (0, 215, 255), 2)
        cv2.circle(frame, action_uv, 5, (0, 215, 255), -1)
        draw_axes(frame, action_pose[:3], action_pose[3:6], intr, length=0.03, thickness=1, alpha=0.6)

    cv2.circle(frame, state_uv, 6, (255, 255, 255), 2)
    draw_axes(frame, state_pose[:3], state_pose[3:6], intr, length=0.045, thickness=2, alpha=1.0)

    text_color = (0, 255, 255) if side == "left" else (255, 180, 0)
    tx = min(state_uv[0] + 12, frame.shape[1] - 380)
    ty = max(state_uv[1] - 12, 70 if side == "left" else 180)

    cv2.putText(
        frame,
        f"{side.upper()} state xyz=({state_pose[0]:+.3f},{state_pose[1]:+.3f},{state_pose[2]:+.3f})",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{side.upper()} state rpy=({state_pose[3]:+.2f},{state_pose[4]:+.2f},{state_pose[5]:+.2f}) grip={state_grip:.3f}",
        (tx, ty + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{side.upper()} action xyz=({action_pose[0]:+.3f},{action_pose[1]:+.3f},{action_pose[2]:+.3f})",
        (tx, ty + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{side.upper()} action rpy=({action_pose[3]:+.2f},{action_pose[4]:+.2f},{action_pose[5]:+.2f}) grip={action_grip:.3f}",
        (tx, ty + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )


def make_overlay_video(dataset_dir, source_seq_dir, episode_index, ego_cam_id, output_path, max_frames):
    info = load_info(dataset_dir)
    video_key = "observation.images.cam_high"
    parquet_path, video_path = locate_episode_paths(dataset_dir, info, episode_index, video_key)
    intr = load_intrinsics(source_seq_dir, ego_cam_id)
    states, actions = load_episode_arrays(parquet_path, info)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(info["features"][video_key]["info"]["video.fps"])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(frame_count, len(states), len(actions))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    H, W = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        cv2.rectangle(frame, (10, 10), (720, 72), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"Episode {episode_index}  Frame {frame_idx + 1}/{total_frames}  FPS {fps:.1f}",
            (22, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "White=current state pose, Yellow=stored action target pose",
            (22, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        draw_hand_overlay(frame, "left", states[frame_idx], actions[frame_idx], intr)
        draw_hand_overlay(frame, "right", states[frame_idx], actions[frame_idx], intr)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Loaded parquet -> {parquet_path}")
    print(f"Loaded video   -> {video_path}")
    print(f"Saved overlay  -> {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to the LeRobot dataset root",
    )
    ap.add_argument(
        "--source_seq_dir",
        required=True,
        help="Path to the original raw sequence directory containing camera_params.json",
    )
    ap.add_argument("--episode_index", type=int, default=0, help="Episode index to inspect")
    ap.add_argument("--ego_cam_id", default="07", help="Camera folder id used as the ego view")
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap for quick debugging")
    ap.add_argument(
        "--output",
        default=None,
        help="Output mp4 path; default is <dataset_dir>/debug_state_video/episode_xxxxxx_state_overlay.mp4",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    source_seq_dir = Path(args.source_seq_dir)
    if args.output is None:
        output_path = dataset_dir / "debug_state_video" / f"episode_{args.episode_index:06d}_state_overlay.mp4"
    else:
        output_path = Path(args.output)

    make_overlay_video(
        dataset_dir,
        source_seq_dir,
        args.episode_index,
        args.ego_cam_id,
        output_path,
        args.max_frames,
    )


if __name__ == "__main__":
    main()
