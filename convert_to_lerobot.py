#!/usr/bin/env python3
"""
Convert ego2 data + extracted wrist poses to LeRobot v2 dataset format.

Directory layout produced:
  <output>/
    meta/info.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    meta/stats.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.ego/episode_000000.mp4
    videos/chunk-000/observation.images.left_hand/episode_000000.mp4
    videos/chunk-000/observation.images.right_hand/episode_000000.mp4

Usage:
    python3 convert_to_lerobot.py \
        --data_dir /home/ubuntu/orbbec/src/sync/test/test/ego2 \
        --poses    /home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/wrist_poses.npz \
        --output   /home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/lerobot_dataset \
        --task     "human ego recovery demonstration"
"""
import argparse, json, subprocess
import numpy as np
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
        "std":  arr.std(axis=0).clip(1e-6).tolist(),
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/ubuntu/orbbec/src/sync/test/test/ego2")
    ap.add_argument("--poses",    default="/home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/wrist_poses.npz")
    ap.add_argument("--output",   default="/home/ubuntu/WorkSpace/ZYC/ego_recovery_data_preprocessing/lerobot_dataset")
    ap.add_argument("--task",     default="human ego recovery demonstration")
    ap.add_argument("--fps",      type=float, default=30.0)
    ap.add_argument("--episode_index", type=int, default=0,
                    help="Episode index (increment when adding more episodes)")
    ap.add_argument("--no_video", action="store_true",
                    help="Skip video encoding (for quick testing)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output)
    fps      = args.fps
    ep_idx   = args.episode_index
    chunk    = ep_idx // 1000

    # ── load poses ────────────────────────────────────────────────────────────
    d = np.load(args.poses)
    left_poses  = d["left_wrist_poses"].astype(np.float32)   # (T,6)
    right_poses = d["right_wrist_poses"].astype(np.float32)  # (T,6)
    if "left_gripper" not in d or "right_gripper" not in d:
        raise ValueError(
            "Pose file is missing left_gripper/right_gripper. "
            "Please re-run extract_wrist_pose.py with the updated script."
        )
    left_gripper = d["left_gripper"].astype(np.float32)
    right_gripper = d["right_gripper"].astype(np.float32)
    T = len(left_poses)
    print(f"Loaded {T} frames of wrist poses")

    left_gripper_state = np.array(
        [map_finger_distance_to_gripper(v) for v in left_gripper],
        dtype=np.float32,
    )
    right_gripper_state = np.array(
        [map_finger_distance_to_gripper(v) for v in right_gripper],
        dtype=np.float32,
    )

    # state: [left pose(6), left gripper(1), right pose(6), right gripper(1)] = 14-D
    states = np.concatenate(
        [
            left_poses,
            left_gripper_state[:, None],
            right_poses,
            right_gripper_state[:, None],
        ],
        axis=1,
    )

    # action: delta pose on smoothed wrist trajectories + next-frame gripper command
    actions = build_human_actions(left_poses, right_poses, left_gripper, right_gripper)

    state_names = (
        [f"left_wrist_{k}"  for k in ["x","y","z","rx","ry","rz"]] +
        ["left_gripper"] +
        [f"right_wrist_{k}" for k in ["x","y","z","rx","ry","rz"]] +
        ["right_gripper"]
    )
    action_names = (
        [f"left_delta_{k}"  for k in ["x","y","z","rx","ry","rz"]] +
        ["left_gripper"] +
        [f"right_delta_{k}" for k in ["x","y","z","rx","ry","rz"]] +
        ["right_gripper"]
    )

    # ── camera info ───────────────────────────────────────────────────────────
    with open(data_dir / "camera_params.json") as f:
        cam_params = json.load(f)
    H = int(cam_params["07"]["RGB"]["intrinsic"]["height"])
    W = int(cam_params["07"]["RGB"]["intrinsic"]["width"])

    # camera_key -> (folder_id, image_subdir)
    cameras = {
        "observation.images.ego":        ("07", "RGB"),
        "observation.images.left_hand":  ("06", "RGB"),
        "observation.images.right_hand": ("08", "RGB"),
    }

    ep_str    = f"episode_{ep_idx:06d}"
    chunk_str = f"chunk-{chunk:03d}"

    # ── encode videos ─────────────────────────────────────────────────────────
    if not args.no_video:
        for video_key, (cam_id, modality) in cameras.items():
            img_dir  = data_dir / cam_id / modality
            vid_path = out_dir / "videos" / chunk_str / video_key / f"{ep_str}.mp4"
            print(f"Encoding {video_key} ...")
            encode_video(img_dir, vid_path, fps)
            print(f"  -> {vid_path}")
    else:
        print("Skipping video encoding (--no_video)")

    # ── build parquet ─────────────────────────────────────────────────────────
    video_type = pa.struct([
        pa.field("path",      pa.string()),
        pa.field("timestamp", pa.float32()),
    ])

    global_offset = ep_idx * T

    # Build column arrays
    col_ep_idx    = pa.array([ep_idx] * T,                    type=pa.int64())
    col_fr_idx    = pa.array(list(range(T)),                  type=pa.int64())
    col_ts        = pa.array([np.float32(t/fps) for t in range(T)], type=pa.float32())
    col_done      = pa.array([False]*(T-1) + [True],          type=pa.bool_())
    col_index     = pa.array(list(range(global_offset, global_offset+T)), type=pa.int64())
    col_task_idx  = pa.array([0] * T,                         type=pa.int64())
    col_state     = pa.array(states.tolist(),                 type=pa.list_(pa.float32()))
    col_action    = pa.array(actions.tolist(),                type=pa.list_(pa.float32()))

    col_videos = {}
    for video_key in cameras:
        vid_rel = f"videos/{chunk_str}/{video_key}/{ep_str}.mp4"
        paths   = [vid_rel] * T
        tss     = [np.float32(t/fps) for t in range(T)]
        col_videos[video_key] = pa.StructArray.from_arrays(
            [pa.array(paths, type=pa.string()), pa.array(tss, type=pa.float32())],
            fields=[pa.field("path", pa.string()), pa.field("timestamp", pa.float32())]
        )

    schema_fields = [
        pa.field("episode_index",      pa.int64()),
        pa.field("frame_index",        pa.int64()),
        pa.field("timestamp",          pa.float32()),
        pa.field("next.done",          pa.bool_()),
        pa.field("index",              pa.int64()),
        pa.field("task_index",         pa.int64()),
        pa.field("observation.state",  pa.list_(pa.float32())),
        pa.field("action",             pa.list_(pa.float32())),
    ] + [pa.field(vk, video_type) for vk in cameras]

    arrays = [
        col_ep_idx, col_fr_idx, col_ts, col_done,
        col_index, col_task_idx, col_state, col_action,
    ] + [col_videos[vk] for vk in cameras]

    table = pa.table(
        {f.name: arr for f, arr in zip(schema_fields, arrays)},
        schema=pa.schema(schema_fields),
    )

    parquet_path = out_dir / "data" / chunk_str / f"{ep_str}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(parquet_path))
    print(f"Parquet -> {parquet_path}  ({T} rows)")

    # ── meta/info.json ────────────────────────────────────────────────────────
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    video_info = {
        "video.fps":          fps,
        "video.codec":        "h264",
        "video.pix_fmt":      "yuv420p",
        "video.is_depth_map": False,
        "has_audio":          False,
    }
    features = {}
    for vk in cameras:
        features[vk] = {
            "dtype":      "video",
            "shape":      [H, W, 3],
            "names":      ["height", "width", "channel"],
            "video_info": video_info,
        }
    features["observation.state"] = {
        "dtype": "float32", "shape": [14], "names": state_names,
    }
    features["action"] = {
        "dtype": "float32", "shape": [14], "names": action_names,
    }
    for col, dtype in [
        ("episode_index","int64"), ("frame_index","int64"),
        ("timestamp","float32"),   ("next.done","bool"),
        ("index","int64"),         ("task_index","int64"),
    ]:
        features[col] = {"dtype": dtype, "shape": [1], "names": None}

    # Read existing info to accumulate episode counts when appending
    info_path = meta_dir / "info.json"
    if info_path.exists() and ep_idx > 0:
        with open(info_path) as f:
            info = json.load(f)
        info["total_episodes"] = ep_idx + 1
        info["total_frames"]   = (ep_idx + 1) * T
        info["total_videos"]   = (ep_idx + 1) * len(cameras)
        info["total_chunks"]   = chunk + 1
        info["splits"]         = {"train": f"0:{ep_idx+1}"}
    else:
        info = {
            "codebase_version": "v2.0",
            "robot_type":       "human",
            "total_episodes":   ep_idx + 1,
            "total_frames":     (ep_idx + 1) * T,
            "total_tasks":      1,
            "total_videos":     (ep_idx + 1) * len(cameras),
            "total_chunks":     chunk + 1,
            "chunks_size":      1000,
            "fps":              fps,
            "splits":           {"train": f"0:{ep_idx+1}"},
            "data_path":        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path":       "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features":         features,
        }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # ── meta/episodes.jsonl ───────────────────────────────────────────────────
    ep_record = {"episode_index": ep_idx, "tasks": [args.task], "length": T}
    mode = "a" if ep_idx > 0 else "w"
    with open(meta_dir / "episodes.jsonl", mode) as f:
        f.write(json.dumps(ep_record) + "\n")

    # ── meta/tasks.jsonl ──────────────────────────────────────────────────────
    tasks_path = meta_dir / "tasks.jsonl"
    if not tasks_path.exists():
        with open(tasks_path, "w") as f:
            f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    # ── meta/stats.json ───────────────────────────────────────────────────────
    stats = {
        "observation.state": compute_stats(states),
        "action":            compute_stats(actions),
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset written to: {out_dir}")
    print(f"  episodes={ep_idx+1}  frames={T}  cameras={len(cameras)}")
    print("\nNext steps:")
    print("  1. Run extract_wrist_pose.py to get wrist_poses.npz")
    print("  2. Run visualize_wrist_pose.py to verify accuracy")
    print("  3. Run this script to build the LeRobot dataset")
    print("  4. For multiple episodes, re-run with --episode_index 1, 2, ...")


if __name__ == "__main__":
    main()
