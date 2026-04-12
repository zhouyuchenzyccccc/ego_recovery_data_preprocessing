#!/usr/bin/env python3
"""
Convert ego data + extracted wrist poses to LeRobot v2 dataset format.

Supports either:
1. A single sequence directory with one pose file, or
2. A parent directory whose immediate children are multiple sequence directories.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = {
    "observation.images.ego":        ("07", "RGB"),
    "observation.images.left_hand":  ("06", "RGB"),
    "observation.images.right_hand": ("08", "RGB"),
}

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


def sort_key(path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_delta(poses):
    """(T,6) absolute -> (T,6) delta; last frame repeats second-to-last."""
    if len(poses) == 0:
        return np.zeros_like(poses)
    if len(poses) == 1:
        return np.zeros_like(poses)
    delta = np.zeros_like(poses)
    delta[:-1] = poses[1:] - poses[:-1]
    delta[:-1, 3:6] = angle_wrap(delta[:-1, 3:6])
    delta[-1] = delta[-2]
    return delta


def map_finger_distance_to_gripper(finger_dist, close_threshold=0.03, open_threshold=0.08):
    """Map thumb-index distance in meters to an ALOHA-like [0, 1] gripper signal."""
    if finger_dist <= close_threshold:
        return np.float32(0.0)
    if finger_dist >= open_threshold:
        return np.float32(1.0)
    value = (finger_dist - close_threshold) / (open_threshold - close_threshold)
    return np.float32(np.clip(value, 0.0, 1.0))


def build_human_actions(left_poses, right_poses, left_gripper, right_gripper):
    """Construct 14-D actions to match the guide: pose delta + next-frame gripper."""
    left_delta = compute_delta(left_poses)
    right_delta = compute_delta(right_poses)
    left_gripper_cmd = np.array(
        [map_finger_distance_to_gripper(v) for v in left_gripper],
        dtype=np.float32,
    )
    right_gripper_cmd = np.array(
        [map_finger_distance_to_gripper(v) for v in right_gripper],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            left_delta,
            left_gripper_cmd[:, None],
            right_delta,
            right_gripper_cmd[:, None],
        ],
        axis=1,
    )


def encode_video(img_dir, out_path, fps):
    """Encode image sequence (00000.jpg ...) to mp4 with libx264."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(img_dir / "%05d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg stderr: {result.stderr[-400:]}")
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def compute_stats(arr):
    """arr: (T, D) float32 -> dict with mean/std/min/max per dim."""
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).clip(1e-6).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
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


def load_sequence_data(data_dir, pose_path):
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
    if not (len(left_poses) == len(right_poses) == len(left_gripper) == len(right_gripper)):
        raise ValueError(f"Pose length mismatch in {pose_path}")

    with open(data_dir / "camera_params.json", encoding="utf-8") as f:
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
        sample_rgb = next((data_dir / "07" / "RGB").glob("*.jpg"), None)
        if sample_rgb is None:
            raise FileNotFoundError(f"No RGB frames found in {data_dir / '07' / 'RGB'}")
        import cv2
        image = cv2.imread(str(sample_rgb))
        if image is None:
            raise FileNotFoundError(f"Failed to read sample RGB frame: {sample_rgb}")
        H, W = image.shape[:2]
    return left_poses, right_poses, left_gripper, right_gripper, H, W


def build_state_action_arrays(left_poses, right_poses, left_gripper, right_gripper):
    left_gripper_state = np.array(
        [map_finger_distance_to_gripper(v) for v in left_gripper],
        dtype=np.float32,
    )
    right_gripper_state = np.array(
        [map_finger_distance_to_gripper(v) for v in right_gripper],
        dtype=np.float32,
    )
    states = np.concatenate(
        [
            left_poses,
            left_gripper_state[:, None],
            right_poses,
            right_gripper_state[:, None],
        ],
        axis=1,
    )
    actions = build_human_actions(left_poses, right_poses, left_gripper, right_gripper)
    return states, actions


def get_feature_names():
    state_names = (
        [f"left_wrist_{k}" for k in ["x", "y", "z", "rx", "ry", "rz"]]
        + ["left_gripper"]
        + [f"right_wrist_{k}" for k in ["x", "y", "z", "rx", "ry", "rz"]]
        + ["right_gripper"]
    )
    action_names = (
        [f"left_delta_{k}" for k in ["x", "y", "z", "rx", "ry", "rz"]]
        + ["left_gripper"]
        + [f"right_delta_{k}" for k in ["x", "y", "z", "rx", "ry", "rz"]]
        + ["right_gripper"]
    )
    return state_names, action_names


def build_features(H, W, fps):
    state_names, action_names = get_feature_names()
    video_info = {
        "video.fps": fps,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    features = {}
    for video_key in CAMERAS:
        features[video_key] = {
            "dtype": "video",
            "shape": [H, W, 3],
            "names": ["height", "width", "channel"],
            "video_info": video_info,
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [14],
        "names": state_names,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [14],
        "names": action_names,
    }
    for col, dtype in [
        ("episode_index", "int64"),
        ("frame_index", "int64"),
        ("timestamp", "float32"),
        ("next.done", "bool"),
        ("index", "int64"),
        ("task_index", "int64"),
    ]:
        features[col] = {"dtype": dtype, "shape": [1], "names": None}
    return features


def write_episode(data_dir, out_dir, fps, ep_idx, task, no_video, global_offset, states, actions):
    T = len(states)
    chunk = ep_idx // 1000
    ep_str = f"episode_{ep_idx:06d}"
    chunk_str = f"chunk-{chunk:03d}"

    if not no_video:
        for video_key, (cam_id, modality) in CAMERAS.items():
            img_dir = data_dir / cam_id / modality
            vid_path = out_dir / "videos" / chunk_str / video_key / f"{ep_str}.mp4"
            print(f"Encoding {video_key} for {data_dir.name} ...")
            encode_video(img_dir, vid_path, fps)
            print(f"  -> {vid_path}")
    else:
        print(f"Skipping video encoding for {data_dir.name} (--no_video)")

    video_type = pa.struct([
        pa.field("path", pa.string()),
        pa.field("timestamp", pa.float32()),
    ])

    columns = {
        "episode_index": pa.array([ep_idx] * T, type=pa.int64()),
        "frame_index": pa.array(list(range(T)), type=pa.int64()),
        "timestamp": pa.array([np.float32(t / fps) for t in range(T)], type=pa.float32()),
        "next.done": pa.array([False] * (T - 1) + [True], type=pa.bool_()),
        "index": pa.array(list(range(global_offset, global_offset + T)), type=pa.int64()),
        "task_index": pa.array([0] * T, type=pa.int64()),
        "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
        "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
    }

    for video_key in CAMERAS:
        vid_rel = f"videos/{chunk_str}/{video_key}/{ep_str}.mp4"
        columns[video_key] = pa.StructArray.from_arrays(
            [
                pa.array([vid_rel] * T, type=pa.string()),
                pa.array([np.float32(t / fps) for t in range(T)], type=pa.float32()),
            ],
            fields=[pa.field("path", pa.string()), pa.field("timestamp", pa.float32())],
        )

    schema = pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float32()),
            pa.field("next.done", pa.bool_()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("observation.state", pa.list_(pa.float32())),
            pa.field("action", pa.list_(pa.float32())),
        ] + [pa.field(video_key, video_type) for video_key in CAMERAS]
    )
    table = pa.table(columns, schema=schema)

    parquet_path = out_dir / "data" / chunk_str / f"{ep_str}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(parquet_path))
    print(f"Parquet -> {parquet_path}  ({T} rows)")

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "episodes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"episode_index": ep_idx, "tasks": [task], "length": T}) + "\n")

    tasks_path = meta_dir / "tasks.jsonl"
    if not tasks_path.exists():
        with open(tasks_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    return T


def write_info(out_dir, features, fps, total_episodes, total_frames):
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = max(1, (total_episodes + 999) // 1000)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "human",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(CAMERAS),
        "total_chunks": total_chunks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


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
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--episode_index", type=int, default=0, help="Starting episode index for this conversion run")
    ap.add_argument("--cam_id", default="07", help="Camera folder used to identify valid sequence directories")
    ap.add_argument("--no_video", action="store_true", help="Skip video encoding (for quick testing)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    seq_dirs = discover_sequence_dirs(data_dir, args.cam_id)
    batch_mode = len(seq_dirs) > 1
    print(f"Discovered {len(seq_dirs)} sequence(s) under {data_dir}")

    meta_dir = out_dir / "meta"
    existing_total_frames = 0
    existing_total_episodes = 0
    info_path = meta_dir / "info.json"
    if info_path.exists() and args.episode_index > 0:
        with open(info_path, encoding="utf-8") as f:
            existing_info = json.load(f)
        existing_total_frames = int(existing_info.get("total_frames", 0))
        existing_total_episodes = int(existing_info.get("total_episodes", 0))

    if args.episode_index == 0:
        for meta_name in ["episodes.jsonl", "tasks.jsonl"]:
            meta_path = meta_dir / meta_name
            if meta_path.exists():
                meta_path.unlink()

    ep_idx = args.episode_index
    global_offset = existing_total_frames
    all_states = []
    all_actions = []
    features = None

    for seq_dir in seq_dirs:
        pose_path = resolve_pose_path(args.poses, seq_dir, batch_mode)
        print(f"Loading poses for {seq_dir.name} from {pose_path}")
        left_poses, right_poses, left_gripper, right_gripper, H, W = load_sequence_data(seq_dir, pose_path)
        states, actions = build_state_action_arrays(left_poses, right_poses, left_gripper, right_gripper)
        if features is None:
            features = build_features(H, W, args.fps)
        T = write_episode(
            seq_dir,
            out_dir,
            args.fps,
            ep_idx,
            args.task,
            args.no_video,
            global_offset,
            states,
            actions,
        )
        ep_idx += 1
        global_offset += T
        all_states.append(states)
        all_actions.append(actions)

    total_episodes = max(existing_total_episodes, ep_idx)
    total_frames = global_offset
    write_info(out_dir, features, args.fps, total_episodes, total_frames)

    stats = {
        "observation.state": compute_stats(np.concatenate(all_states, axis=0)),
        "action": compute_stats(np.concatenate(all_actions, axis=0)),
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset written to: {out_dir}")
    print(f"  newly_added_episodes={len(seq_dirs)}  total_episodes={total_episodes}  total_frames={total_frames}")


if __name__ == "__main__":
    main()
