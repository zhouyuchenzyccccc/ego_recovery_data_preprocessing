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
import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


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


def create_detector(model_path):
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detect_wrist_points(detector, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    H, W = frame_bgr.shape[:2]
    wrists = {}
    for lms, handedness in zip(result.hand_landmarks, result.handedness):
        side = handedness[0].category_name.lower()
        side = "right" if side == "left" else "left"
        lm0 = lms[0]
        u = int(np.clip(lm0.x * W, 0, W - 1))
        v = int(np.clip(lm0.y * H, 0, H - 1))
        wrists[side] = (u, v)
    return wrists


def compute_arrow_scale(actions, expected, layout):
    all_xy = []
    for side_key in ["left_action_pose", "right_action_pose"]:
        action_slice = layout[side_key]
        all_xy.append(actions[:, action_slice][:, :2])
        all_xy.append(expected[:, action_slice][:, :2])
    all_xy = np.concatenate(all_xy, axis=0)
    scale_ref = float(np.percentile(np.abs(all_xy), 95))
    return 70.0 / max(scale_ref, 1e-4)


def draw_arrow(frame, origin, dx, dy, scale, color, label, y_offset):
    ox, oy = origin
    end = (int(round(ox + dx * scale)), int(round(oy + dy * scale)))
    cv2.arrowedLine(frame, (ox, oy), end, color, 2, tipLength=0.25)
    cv2.putText(
        frame,
        label,
        (ox + 10, oy + y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_hand_action_overlay(
    frame,
    side,
    wrist_point,
    stored_pose_action,
    expected_pose_action,
    action_error,
    pose_pred_error,
    arrow_scale,
):
    color_stored = (0, 165, 255)
    color_expected = (60, 200, 60)
    color_anchor = (255, 255, 255)
    ox, oy = wrist_point
    cv2.circle(frame, wrist_point, 6, color_anchor, 2)
    cv2.putText(
        frame,
        side.upper(),
        (ox - 18, oy - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_anchor,
        1,
        cv2.LINE_AA,
    )

    draw_arrow(
        frame,
        wrist_point,
        float(stored_pose_action[0]),
        float(stored_pose_action[1]),
        arrow_scale,
        color_stored,
        f"S dx={stored_pose_action[0]:+.3f} dy={stored_pose_action[1]:+.3f}",
        18,
    )
    draw_arrow(
        frame,
        wrist_point,
        float(expected_pose_action[0]),
        float(expected_pose_action[1]),
        arrow_scale,
        color_expected,
        f"R dx={expected_pose_action[0]:+.3f} dy={expected_pose_action[1]:+.3f}",
        36,
    )

    cv2.putText(
        frame,
        f"dz S/R {stored_pose_action[2]:+.3f}/{expected_pose_action[2]:+.3f}",
        (ox + 10, oy + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"rot S {stored_pose_action[3]:+.2f},{stored_pose_action[4]:+.2f},{stored_pose_action[5]:+.2f}",
        (ox + 10, oy + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color_stored,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"rot R {expected_pose_action[3]:+.2f},{expected_pose_action[4]:+.2f},{expected_pose_action[5]:+.2f}",
        (ox + 10, oy + 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color_expected,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"err={action_error:.5f} next_pose_err={pose_pred_error:.5f}",
        (ox + 10, oy + 107),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 140, 255) if action_error > 1e-4 else (60, 200, 60),
        1,
        cv2.LINE_AA,
    )


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


def draw_global_legend(frame, frame_idx, total_frames, fps, frame_mae, frame_max):
    cv2.rectangle(frame, (12, 12), (540, 92), (20, 20, 20), -1)
    cv2.putText(frame, f"Frame {frame_idx + 1}/{total_frames}  FPS {fps:.1f}", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Orange=stored action   Green=recomputed-from-state", (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"frame MAE={frame_mae:.6f}   frame max err={frame_max:.6f}", (24, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def make_overlay_video(dataset_dir, info, episode_index, video_key, output_path, max_frames, model_path):
    parquet_path, video_path = locate_episode_paths(dataset_dir, info, episode_index, video_key)
    states, actions, state_names, action_names = load_episode_arrays(parquet_path, info)
    layout = infer_layout(state_names, action_names)
    expected = recompute_expected_actions(states, layout)
    errors = np.abs(actions - expected)
    left_pred_err, right_pred_err = compute_prediction_error(states, actions, layout)
    arrow_scale = compute_arrow_scale(actions, expected, layout)
    detector = create_detector(model_path)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_wrist_points = {}
    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        wrist_points = detect_wrist_points(detector, frame)
        last_wrist_points.update(wrist_points)
        frame_mae = float(errors[frame_idx].mean())
        frame_max = float(errors[frame_idx].max())
        draw_global_legend(frame, frame_idx, total_frames, fps, frame_mae, frame_max)

        for side, pose_slice, pred_err in [
            ("left", layout["left_action_pose"], left_pred_err),
            ("right", layout["right_action_pose"], right_pred_err),
        ]:
            wrist_point = wrist_points.get(side) or last_wrist_points.get(side)
            if wrist_point is None:
                continue
            draw_hand_action_overlay(
                frame,
                side,
                wrist_point,
                actions[frame_idx, pose_slice],
                expected[frame_idx, pose_slice],
                float(np.mean(errors[frame_idx, pose_slice])),
                float(pred_err[frame_idx]),
                arrow_scale,
            )

        writer.write(frame)

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
        "--model_path",
        default="/home/ubuntu/WorkSpace/ZYC/hamer/_DATA/mediapipe/hand_landmarker.task",
        help="MediaPipe hand_landmarker.task path for wrist localization on video frames",
    )
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

    make_overlay_video(
        dataset_dir,
        info,
        args.episode_index,
        args.video_key,
        output_path,
        args.max_frames,
        args.model_path,
    )


if __name__ == "__main__":
    main()
