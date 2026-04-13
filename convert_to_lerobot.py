#!/usr/bin/env python3
"""
Convert human ego-hand data to LeRobot v2.0 format with robot-aligned schema.

Key design choices:
1. `observation.state` is absolute 14-D wrist pose + gripper state.
2. `action` is also absolute 14-D target pose, stored as next-frame absolute pose.
3. All resampling happens in absolute pose space.
4. Videos are resampled to target FPS with nearest-neighbor frame repetition.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TARGET_FPS_DEFAULT = 50.0
CHUNKS_SIZE = 1000
ROBOT_TYPE = "human"
VIDEO_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_CRF = "18"

CAMERAS = {
    "observation.images.cam_high": ("07", "RGB"),
    "observation.images.cam_left_wrist": ("06", "RGB"),
    "observation.images.cam_right_wrist": ("08", "RGB"),
}

STATE_NAMES = [
    "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper",
    "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
]


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


def compute_stats(arr):
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).clip(1e-6).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
    }


def image_stats():
    return {
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.5]], [[0.5]], [[0.5]]],
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
    }


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


def resolve_pose_path(poses_arg, seq_dir, batch_mode):
    if poses_arg is None:
        pose_path = seq_dir / "wrist_poses.npz"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing pose file: {pose_path}")
        return pose_path

    poses_path = Path(poses_arg)
    if not batch_mode and poses_path.suffix == ".npz":
        return poses_path

    candidates = [
        poses_path / f"{seq_dir.name}_wrist_poses.npz",
        poses_path / seq_dir.name / "wrist_poses.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find pose file for sequence {seq_dir.name} under {poses_path}. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def map_finger_distance_to_gripper(finger_dist, close_threshold=0.03, open_threshold=0.08):
    if finger_dist <= close_threshold:
        return np.float32(0.0)
    if finger_dist >= open_threshold:
        return np.float32(1.0)
    value = (finger_dist - close_threshold) / (open_threshold - close_threshold)
    return np.float32(np.clip(value, 0.0, 1.0))


def maybe_convert_mm_to_m(poses, seq_name):
    poses = poses.copy()
    pos = poses[:, :3]
    abs_pos_p95 = float(np.percentile(np.abs(pos), 95))
    delta_p95 = float(np.percentile(np.abs(np.diff(pos, axis=0)), 95)) if len(pos) > 1 else 0.0
    if abs_pos_p95 > 10.0 or delta_p95 > 1.0:
        poses[:, :3] /= 1000.0
        print(
            f"[unit-fix] Sequence {seq_name}: detected likely millimeter positions "
            f"(p95 abs={abs_pos_p95:.3f}, p95 delta={delta_p95:.3f}), converted to meters."
        )
    return poses


def build_absolute_state(left_poses, right_poses, left_gripper, right_gripper, seq_name):
    left_poses = maybe_convert_mm_to_m(left_poses, f"{seq_name}/left")
    right_poses = maybe_convert_mm_to_m(right_poses, f"{seq_name}/right")
    left_gripper = np.array([map_finger_distance_to_gripper(v) for v in left_gripper], dtype=np.float32)
    right_gripper = np.array([map_finger_distance_to_gripper(v) for v in right_gripper], dtype=np.float32)
    return np.concatenate(
        [
            left_poses[:, :6],
            left_gripper[:, None],
            right_poses[:, :6],
            right_gripper[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def unwrap_pose_angles(states):
    states = states.copy()
    states[:, 3:6] = np.unwrap(states[:, 3:6], axis=0)
    states[:, 10:13] = np.unwrap(states[:, 10:13], axis=0)
    return states


def resample_absolute_states(states, source_fps, target_fps):
    source_len = len(states)
    if source_len == 0:
        return states
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")

    target_len = max(1, int(round(source_len * target_fps / source_fps)))
    if target_len == source_len and abs(source_fps - target_fps) < 1e-8:
        return states.astype(np.float32)

    states_unwrapped = unwrap_pose_angles(states)
    source_times = np.arange(source_len, dtype=np.float64) / source_fps
    target_times = np.arange(target_len, dtype=np.float64) / target_fps

    resampled = np.zeros((target_len, states.shape[1]), dtype=np.float32)
    for dim in range(states.shape[1]):
        resampled[:, dim] = np.interp(target_times, source_times, states_unwrapped[:, dim]).astype(np.float32)
    return resampled


def build_absolute_actions(states):
    actions = np.zeros_like(states, dtype=np.float32)
    if len(states) == 0:
        return actions
    if len(states) == 1:
        actions[0] = states[0]
        return actions
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


def load_sequence_data(seq_dir, pose_path, fallback_source_fps):
    pose_data = np.load(pose_path)
    left_poses = pose_data["left_wrist_poses"].astype(np.float32)
    right_poses = pose_data["right_wrist_poses"].astype(np.float32)
    if "left_gripper" not in pose_data or "right_gripper" not in pose_data:
        raise ValueError(
            f"Pose file is missing left_gripper/right_gripper: {pose_path}. "
            "Please re-run extract_wrist_pose.py with the updated script."
        )
    left_gripper = pose_data["left_gripper"].astype(np.float32)
    right_gripper = pose_data["right_gripper"].astype(np.float32)
    source_fps = float(pose_data["fps"]) if "fps" in pose_data else float(fallback_source_fps)
    if not (len(left_poses) == len(right_poses) == len(left_gripper) == len(right_gripper)):
        raise ValueError(f"Pose length mismatch in {pose_path}")

    with open(seq_dir / "camera_params.json", encoding="utf-8") as f:
        cam_params = json.load(f)
    cam = get_by_alias(cam_params, ["07", "7"])
    rgb_intr = (
        normalize_intrinsic(get_by_alias(cam, ["RGB", "rgb", "color", "Color"]))
        or normalize_intrinsic(get_by_alias(cam, ["rgb_intrinsic", "color_intrinsic"]))
        or normalize_intrinsic(cam)
    )
    if rgb_intr is not None and "height" in rgb_intr and "width" in rgb_intr:
        H = int(rgb_intr["height"])
        W = int(rgb_intr["width"])
    else:
        sample_rgb = next((seq_dir / "07" / "RGB").glob("*.jpg"), None)
        if sample_rgb is None:
            raise FileNotFoundError(f"No RGB frames found in {seq_dir / '07' / 'RGB'}")
        image = cv2.imread(str(sample_rgb))
        if image is None:
            raise FileNotFoundError(f"Failed to read sample RGB frame: {sample_rgb}")
        H, W = image.shape[:2]

    states = build_absolute_state(left_poses, right_poses, left_gripper, right_gripper, seq_dir.name)
    states = resample_absolute_states(states, source_fps, TARGET_FPS_DEFAULT)
    actions = build_absolute_actions(states)
    target_len = len(states)
    return states, actions, H, W, source_fps, target_len


def build_target_frame_indices(source_len, source_fps, target_len, target_fps):
    if source_len == 0:
        return np.array([], dtype=np.int64)
    source_times = np.arange(source_len, dtype=np.float64) / source_fps
    target_times = np.arange(target_len, dtype=np.float64) / target_fps
    indices = np.searchsorted(source_times, target_times, side="left")
    indices = np.clip(indices, 0, source_len - 1)
    prev_indices = np.clip(indices - 1, 0, source_len - 1)
    prev_dist = np.abs(target_times - source_times[prev_indices])
    next_dist = np.abs(target_times - source_times[indices])
    use_prev = prev_dist <= next_dist
    return np.where(use_prev, prev_indices, indices).astype(np.int64)


def write_video_from_paths(frame_paths, selected_indices, out_path, fps):
    if len(selected_indices) == 0:
        raise ValueError("selected_indices is empty")
    first_frame = cv2.imread(str(frame_paths[int(selected_indices[0])]))
    if first_frame is None:
        raise FileNotFoundError(f"Failed to read frame: {frame_paths[int(selected_indices[0])]}")
    H, W = first_frame.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", VIDEO_CODEC,
        "-pix_fmt", VIDEO_PIX_FMT,
        "-crf", VIDEO_CRF,
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for idx in selected_indices:
            frame = cv2.imread(str(frame_paths[int(idx)]))
            if frame is None:
                raise FileNotFoundError(f"Failed to read frame: {frame_paths[int(idx)]}")
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}) for {out_path}")


def write_episode(seq_dir, out_dir, episode_index, task, states, actions, fps, source_fps, target_len, global_offset, no_video):
    chunk_idx = episode_index // CHUNKS_SIZE
    chunk_str = f"chunk-{chunk_idx:03d}"
    ep_str = f"episode_{episode_index:06d}"

    if not no_video:
        for video_key, (cam_id, modality) in CAMERAS.items():
            frame_paths = sorted((seq_dir / cam_id / modality).glob("*.jpg"))
            if not frame_paths:
                raise FileNotFoundError(f"No frames found in {seq_dir / cam_id / modality}")
            selected_indices = build_target_frame_indices(len(frame_paths), source_fps, target_len, fps)
            video_path = out_dir / "videos" / chunk_str / video_key / f"{ep_str}.mp4"
            print(f"Encoding {video_key} for {seq_dir.name} ...")
            write_video_from_paths(frame_paths, selected_indices, video_path, fps)
            print(f"  -> {video_path}")
    else:
        print(f"Skipping video encoding for {seq_dir.name} (--no_video)")

    parquet_dir = out_dir / "data" / chunk_str
    parquet_dir.mkdir(parents=True, exist_ok=True)
    T = len(states)
    timestamps = np.arange(T, dtype=np.float64) / fps
    frame_indices = np.arange(T, dtype=np.int64)
    episode_indices = np.full(T, episode_index, dtype=np.int64)
    global_indices = np.arange(global_offset, global_offset + T, dtype=np.int64)
    task_indices = np.zeros(T, dtype=np.int64)

    table = pa.table({
        "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
        "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
        "timestamp": pa.array(timestamps),
        "frame_index": pa.array(frame_indices),
        "episode_index": pa.array(episode_indices),
        "index": pa.array(global_indices),
        "task_index": pa.array(task_indices),
    })
    parquet_path = parquet_dir / f"{ep_str}.parquet"
    pq.write_table(table, parquet_path)
    print(f"Parquet -> {parquet_path}  ({T} rows)")
    return T


def build_info(H, W, total_episodes, total_frames, fps):
    total_chunks = max(1, (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE)

    def video_feature():
        return {
            "dtype": "video",
            "shape": [3, H, W],
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": float(fps),
                "video.height": H,
                "video.width": W,
                "video.channels": 3,
                "video.codec": VIDEO_CODEC,
                "video.pix_fmt": VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "robot_type": ROBOT_TYPE,
        "fps": float(fps),
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [14],
                "names": STATE_NAMES,
            },
            **{video_key: video_feature() for video_key in CAMERAS},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "splits": {"train": f"0:{total_episodes}"},
        "license": "apache-2.0",
        "tags": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Single sequence dir or a parent dir containing multiple sequence subdirectories",
    )
    ap.add_argument(
        "--poses",
        default=None,
        help="Single .npz path, or a directory containing batch pose files. "
             "If omitted, each sequence is expected to contain wrist_poses.npz.",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--task", default="human ego recovery demonstration")
    ap.add_argument("--source_fps", type=float, default=30.0, help="Fallback source FPS if pose file does not contain fps")
    ap.add_argument("--target_fps", type=float, default=TARGET_FPS_DEFAULT, help="Target FPS, should match robot data")
    ap.add_argument("--episode_index", type=int, default=0, help="Starting episode index for this conversion run")
    ap.add_argument("--cam_id", default="07", help="Camera folder used to identify valid sequence directories")
    ap.add_argument("--no_video", action="store_true", help="Skip video encoding (for quick testing)")
    args = ap.parse_args()

    if abs(args.target_fps - TARGET_FPS_DEFAULT) > 1e-8:
        raise ValueError(f"This script is configured to output 50 FPS. Received target_fps={args.target_fps}.")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    seq_dirs = discover_sequence_dirs(data_dir, args.cam_id)
    batch_mode = len(seq_dirs) > 1
    print(f"Discovered {len(seq_dirs)} sequence(s) under {data_dir}")

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.episode_index == 0:
        for meta_name in ["episodes.jsonl", "tasks.jsonl"]:
            meta_path = meta_dir / meta_name
            if meta_path.exists():
                meta_path.unlink()

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    episodes_meta = []
    total_frames = 0
    all_states = []
    all_actions = []
    info_H = None
    info_W = None

    for seq_offset, seq_dir in enumerate(seq_dirs):
        episode_index = args.episode_index + seq_offset
        pose_path = resolve_pose_path(args.poses, seq_dir, batch_mode)
        print(f"Loading poses for {seq_dir.name} from {pose_path}")
        states, actions, H, W, source_fps, target_len = load_sequence_data(seq_dir, pose_path, args.source_fps)
        if info_H is None:
            info_H, info_W = H, W

        written_frames = write_episode(
            seq_dir,
            out_dir,
            episode_index,
            args.task,
            states,
            actions,
            args.target_fps,
            source_fps,
            target_len,
            total_frames,
            args.no_video,
        )
        episodes_meta.append({"episode_index": episode_index, "tasks": [args.task], "length": written_frames})
        total_frames += written_frames
        all_states.append(states)
        all_actions.append(actions)

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for episode in episodes_meta:
            f.write(json.dumps(episode) + "\n")

    stats = {
        "observation.state": compute_stats(np.concatenate(all_states, axis=0)),
        "action": compute_stats(np.concatenate(all_actions, axis=0)),
        **{video_key: image_stats() for video_key in CAMERAS},
    }
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    info = build_info(info_H, info_W, len(episodes_meta), total_frames, args.target_fps)
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset written to: {out_dir}")
    print(f"  episodes={len(episodes_meta)}  total_frames={total_frames}  fps={args.target_fps}")


if __name__ == "__main__":
    main()
