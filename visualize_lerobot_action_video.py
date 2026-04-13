#!/usr/bin/env python3
"""
Overlay LeRobot action diagnostics directly on episode videos.

The output video keeps the original camera view on the left and appends
an action diagnostics panel on the right. For every frame it shows:
1. stored action from the parquet,
2. re-computed action from state deltas,
3. per-dimension absolute error,
4. frame-level agreement summary.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def angle_wrap(values):
    return (values + np.pi) % (2 * np.pi) - np.pi


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


def infer_layout(state_names, action_names):
    state_dim = len(state_names)
    action_dim = len(action_names)
    has_gripper = (
        state_dim == 14
        and action_dim == 14
        and "left_gripper" in state_names
        and "right_gripper" in state_names
    )
    if has_gripper:
        return {
            "has_gripper": True,
            "left_pose": slice(0, 6),
            "left_gripper": 6,
            "right_pose": slice(7, 13),
            "right_gripper": 13,
            "left_action_pose": slice(0, 6),
            "left_action_gripper": 6,
            "right_action_pose": slice(7, 13),
            "right_action_gripper": 13,
        }
    if state_dim == 12 and action_dim == 12:
        return {
            "has_gripper": False,
            "left_pose": slice(0, 6),
            "right_pose": slice(6, 12),
            "left_action_pose": slice(0, 6),
            "right_action_pose": slice(6, 12),
        }
    raise ValueError(
        f"Unsupported dims: state={state_dim}, action={action_dim}. "
        "Expected either 12/12 or 14/14."
    )


def recompute_expected_actions(states, layout):
    action_dim = 14 if layout["has_gripper"] else 12
    expected = np.zeros((len(states), action_dim), dtype=np.float32)

    left_pose = states[:, layout["left_pose"]]
    right_pose = states[:, layout["right_pose"]]

    left_delta = np.zeros_like(left_pose)
    right_delta = np.zeros_like(right_pose)
    if len(states) > 1:
        left_delta[:-1] = left_pose[1:] - left_pose[:-1]
        right_delta[:-1] = right_pose[1:] - right_pose[:-1]
        left_delta[:-1, 3:6] = angle_wrap(left_delta[:-1, 3:6])
        right_delta[:-1, 3:6] = angle_wrap(right_delta[:-1, 3:6])
        left_delta[-1] = left_delta[-2]
        right_delta[-1] = right_delta[-2]

    expected[:, layout["left_action_pose"]] = left_delta
    expected[:, layout["right_action_pose"]] = right_delta

    if layout["has_gripper"]:
        expected[:, layout["left_action_gripper"]] = states[:, layout["left_gripper"]]
        expected[:, layout["right_action_gripper"]] = states[:, layout["right_gripper"]]

    return expected


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
    return states, actions, state_names, action_names


def compute_prediction_error(states, actions, layout):
    left_pose = states[:, layout["left_pose"]]
    right_pose = states[:, layout["right_pose"]]
    pred_left = left_pose.copy()
    pred_right = right_pose.copy()
    pred_left[:-1] = left_pose[:-1] + actions[:-1, layout["left_action_pose"]]
    pred_right[:-1] = right_pose[:-1] + actions[:-1, layout["right_action_pose"]]
    pred_left[:-1, 3:6] = left_pose[:-1, 3:6] + actions[:-1, layout["left_action_pose"]][:, 3:6]
    pred_right[:-1, 3:6] = right_pose[:-1, 3:6] + actions[:-1, layout["right_action_pose"]][:, 3:6]

    left_err = np.zeros(len(states), dtype=np.float32)
    right_err = np.zeros(len(states), dtype=np.float32)
    if len(states) > 1:
        left_diff = pred_left[:-1] - left_pose[1:]
        right_diff = pred_right[:-1] - right_pose[1:]
        left_diff[:, 3:6] = angle_wrap(left_diff[:, 3:6])
        right_diff[:, 3:6] = angle_wrap(right_diff[:, 3:6])
        left_err[:-1] = np.linalg.norm(left_diff, axis=1)
        right_err[:-1] = np.linalg.norm(right_diff, axis=1)
        left_err[-1] = left_err[-2]
        right_err[-1] = right_err[-2]
    return left_err, right_err


def draw_bar(panel, x0, y0, width, height, value, scale, color, label):
    cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (70, 70, 70), 1)
    center = x0 + width // 2
    cv2.line(panel, (center, y0), (center, y0 + height), (90, 90, 90), 1)
    if scale < 1e-8:
        filled = center
    else:
        delta = int((value / scale) * (width // 2 - 4))
        filled = center + delta
    if filled >= center:
        cv2.rectangle(panel, (center, y0 + 2), (filled, y0 + height - 2), color, -1)
    else:
        cv2.rectangle(panel, (filled, y0 + 2), (center, y0 + height - 2), color, -1)
    cv2.putText(
        panel,
        label,
        (x0, y0 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )


def draw_header(panel, frame_idx, total_frames, fps, frame_mae, frame_max, left_pred_err, right_pred_err):
    cv2.putText(panel, f"Frame {frame_idx + 1}/{total_frames}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"FPS {fps:.1f}", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    status = "MATCH" if frame_max < 1e-4 else "CHECK"
    status_color = (40, 180, 40) if status == "MATCH" else (0, 140, 255)
    cv2.rectangle(panel, (360, 16), (520, 58), status_color, -1)
    cv2.putText(panel, status, (393, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(panel, f"frame MAE: {frame_mae:.6f}", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, f"frame max err: {frame_max:.6f}", (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, f"pred next err L/R: {left_pred_err:.5f} / {right_pred_err:.5f}", (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)


def render_panel(frame_idx, total_frames, fps, action_names, actions, expected, errors, scales, left_pred_err, right_pred_err):
    panel_h = 150 + len(action_names) * 34
    panel = np.full((panel_h, 620, 3), 24, dtype=np.uint8)

    frame_mae = float(errors[frame_idx].mean())
    frame_max = float(errors[frame_idx].max())
    draw_header(
        panel,
        frame_idx,
        total_frames,
        fps,
        frame_mae,
        frame_max,
        float(left_pred_err[frame_idx]),
        float(right_pred_err[frame_idx]),
    )

    y = 170
    for dim_idx, name in enumerate(action_names):
        scale = scales[dim_idx]
        stored = float(actions[frame_idx, dim_idx])
        recomputed = float(expected[frame_idx, dim_idx])
        err = float(errors[frame_idx, dim_idx])

        cv2.putText(panel, f"{name}", (18, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1, cv2.LINE_AA)
        draw_bar(panel, 180, y - 22, 170, 14, stored, scale, (220, 120, 30), f"stored {stored:+.4f}")
        draw_bar(panel, 360, y - 22, 170, 14, recomputed, scale, (50, 190, 70), f"recomp {recomputed:+.4f}")
        err_color = (50, 190, 70) if err < 1e-5 else (0, 140, 255) if err < 1e-3 else (40, 40, 220)
        cv2.putText(panel, f"err={err:.6f}", (540, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, err_color, 1, cv2.LINE_AA)
        y += 34

    return panel


def make_overlay_video(dataset_dir, info, episode_index, video_key, output_path, max_frames):
    parquet_path, video_path = locate_episode_paths(dataset_dir, info, episode_index, video_key)
    states, actions, state_names, action_names = load_episode_arrays(parquet_path, info)
    layout = infer_layout(state_names, action_names)
    expected = recompute_expected_actions(states, layout)
    errors = np.abs(actions - expected)
    left_pred_err, right_pred_err = compute_prediction_error(states, actions, layout)
    scales = np.maximum(np.max(np.abs(np.concatenate([actions, expected], axis=0)), axis=0), 1e-6)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(info["features"][video_key]["video_info"]["video.fps"])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(len(actions), frame_count)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    frame_h, frame_w = first_frame.shape[:2]
    panel = render_panel(0, total_frames, fps, action_names, actions, expected, errors, scales, left_pred_err, right_pred_err)
    panel_h, panel_w = panel.shape[:2]

    canvas_h = max(frame_h, panel_h)
    canvas_w = frame_w + panel_w
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        panel = render_panel(
            frame_idx,
            total_frames,
            fps,
            action_names,
            actions,
            expected,
            errors,
            scales,
            left_pred_err,
            right_pred_err,
        )

        canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
        canvas[:frame_h, :frame_w] = frame
        canvas[:panel.shape[0], frame_w:frame_w + panel_w] = panel
        cv2.putText(
            canvas,
            f"Video: {video_key}",
            (18, canvas_h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)

    cap.release()
    writer.release()

    summary = {
        "episode_index": episode_index,
        "video_key": video_key,
        "num_frames_used": total_frames,
        "overall_mae": float(errors[:total_frames].mean()),
        "overall_max_abs_error": float(errors[:total_frames].max()),
        "left_pred_next_pose_error_mean": float(left_pred_err[:total_frames].mean()),
        "right_pred_next_pose_error_mean": float(right_pred_err[:total_frames].mean()),
        "action_names": action_names,
    }
    summary_path = output_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loaded parquet -> {parquet_path}")
    print(f"Loaded video   -> {video_path}")
    print(f"Saved overlay  -> {output_path}")
    print(f"Saved summary  -> {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        default="/home/ubuntu/orbbec/src/sync/collection_000/lerobot_dataset",
        help="Path to the LeRobot dataset root",
    )
    ap.add_argument("--episode_index", type=int, default=0, help="Episode index to inspect")
    ap.add_argument(
        "--video_key",
        default="observation.images.ego",
        choices=["observation.images.ego", "observation.images.left_hand", "observation.images.right_hand"],
        help="Which video stream to overlay diagnostics onto",
    )
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap for quick debugging")
    ap.add_argument(
        "--output",
        default=None,
        help="Output mp4 path; default is <dataset_dir>/debug_action_video/<episode>_<video_key>.mp4",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    info = load_info(dataset_dir)
    if args.output is None:
        safe_video_key = args.video_key.replace(".", "_")
        output_path = dataset_dir / "debug_action_video" / f"episode_{args.episode_index:06d}_{safe_video_key}.mp4"
    else:
        output_path = Path(args.output)

    make_overlay_video(dataset_dir, info, args.episode_index, args.video_key, output_path, args.max_frames)


if __name__ == "__main__":
    main()
