#!/usr/bin/env python3
"""
Visualize whether actions stored in a LeRobot dataset match the state deltas.

The script:
1. loads one episode parquet from a LeRobot dataset,
2. re-computes pose deltas from observation.state,
3. compares them against the stored action,
4. saves diagnostic plots and a JSON summary.

It supports both:
- 12-D human data: left/right wrist 6-DoF state + 12-D delta action
- 14-D human data: left/right wrist 6-DoF + gripper state/action
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def angle_wrap(values):
    return (values + np.pi) % (2 * np.pi) - np.pi


def load_info(dataset_dir):
    info_path = Path(dataset_dir) / "meta" / "info.json"
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def locate_parquet(dataset_dir, info, episode_index):
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk = episode_index // int(info.get("chunks_size", 1000))
    rel_path = data_path.format(episode_chunk=chunk, episode_index=episode_index)
    parquet_path = Path(dataset_dir) / rel_path
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    return parquet_path


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
        f"Unsupported state/action dims: state={state_dim}, action={action_dim}. "
        "Expected either 12/12 or 14/14."
    )


def recompute_expected_actions(states, layout):
    T = len(states)
    expected = np.zeros_like(states[:, :12] if not layout["has_gripper"] else states[:, :14], dtype=np.float32)

    left_pose = states[:, layout["left_pose"]]
    right_pose = states[:, layout["right_pose"]]

    left_delta = np.zeros_like(left_pose)
    right_delta = np.zeros_like(right_pose)

    if T > 1:
        left_delta[:-1] = left_pose[1:] - left_pose[:-1]
        right_delta[:-1] = right_pose[1:] - right_pose[:-1]
        left_delta[:-1, 3:6] = angle_wrap(left_delta[:-1, 3:6])
        right_delta[:-1, 3:6] = angle_wrap(right_delta[:-1, 3:6])
        left_delta[-1] = left_delta[-2]
        right_delta[-1] = right_delta[-2]

    if layout["has_gripper"]:
        expected[:, layout["left_action_pose"]] = left_delta
        expected[:, layout["right_action_pose"]] = right_delta
        expected[:, layout["left_action_gripper"]] = states[:, layout["left_gripper"]]
        expected[:, layout["right_action_gripper"]] = states[:, layout["right_gripper"]]
    else:
        expected[:, layout["left_action_pose"]] = left_delta
        expected[:, layout["right_action_pose"]] = right_delta

    return expected


def compare_gripper_alignment(states, actions, layout):
    if not layout["has_gripper"]:
        return None
    current_left = np.abs(actions[:, layout["left_action_gripper"]] - states[:, layout["left_gripper"]]).mean()
    current_right = np.abs(actions[:, layout["right_action_gripper"]] - states[:, layout["right_gripper"]]).mean()

    next_left = current_left
    next_right = current_right
    if len(states) > 1:
        next_left = np.abs(actions[:-1, layout["left_action_gripper"]] - states[1:, layout["left_gripper"]]).mean()
        next_right = np.abs(actions[:-1, layout["right_action_gripper"]] - states[1:, layout["right_gripper"]]).mean()

    return {
        "current_frame_mae": {
            "left": float(current_left),
            "right": float(current_right),
        },
        "next_frame_mae": {
            "left": float(next_left),
            "right": float(next_right),
        },
        "more_likely_alignment": "current"
        if current_left + current_right <= next_left + next_right
        else "next",
    }


def save_action_plot(actions, expected, action_names, out_path, title):
    dims = len(action_names)
    cols = 2
    rows = int(np.ceil(dims / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3 * rows, 4)), squeeze=False)
    x = np.arange(len(actions))
    for idx, name in enumerate(action_names):
        ax = axes[idx // cols][idx % cols]
        ax.plot(x, actions[:, idx], label="stored", linewidth=1.4)
        ax.plot(x, expected[:, idx], label="recomputed", linewidth=1.0, alpha=0.85)
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend()
    for idx in range(dims, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_error_plot(errors, action_names, out_path, title):
    dims = len(action_names)
    cols = 2
    rows = int(np.ceil(dims / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3 * rows, 4)), squeeze=False)
    x = np.arange(len(errors))
    for idx, name in enumerate(action_names):
        ax = axes[idx // cols][idx % cols]
        ax.plot(x, errors[:, idx], color="crimson", linewidth=1.1)
        ax.set_title(f"{name} abs error")
        ax.grid(True, alpha=0.25)
    for idx in range(dims, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_norm_plot(actions, expected, out_path, title):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    action_norm = np.linalg.norm(actions, axis=1)
    expected_norm = np.linalg.norm(expected, axis=1)
    error_norm = np.linalg.norm(actions - expected, axis=1)

    axes[0].plot(action_norm, label="stored action norm", linewidth=1.4)
    axes[0].plot(expected_norm, label="recomputed action norm", linewidth=1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Action norm")

    axes[1].plot(error_norm, color="crimson", linewidth=1.1)
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title("Action error norm")
    axes[1].set_xlabel("Frame")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(actions, expected, action_names, fps, layout, gripper_check):
    errors = np.abs(actions - expected)
    per_dim = []
    for idx, name in enumerate(action_names):
        per_dim.append(
            {
                "name": name,
                "mae": float(errors[:, idx].mean()),
                "max_abs_error": float(errors[:, idx].max()),
                "stored_mean": float(actions[:, idx].mean()),
                "recomputed_mean": float(expected[:, idx].mean()),
            }
        )

    return {
        "fps": fps,
        "action_dim": int(actions.shape[1]),
        "num_frames": int(actions.shape[0]),
        "has_gripper": bool(layout["has_gripper"]),
        "overall_mae": float(errors.mean()),
        "overall_max_abs_error": float(errors.max()),
        "per_dim": per_dim,
        "gripper_alignment_check": gripper_check,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        default="/home/ubuntu/orbbec/src/sync/collection_000/lerobot_dataset",
        help="Path to the LeRobot dataset root",
    )
    ap.add_argument("--episode_index", type=int, default=0, help="Episode index to inspect")
    ap.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save plots and summary; default is <dataset_dir>/debug_action_viz",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    info = load_info(dataset_dir)
    parquet_path = locate_parquet(dataset_dir, info, args.episode_index)

    try:
        table = pq.ParquetFile(parquet_path).read()
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read parquet file {parquet_path}. "
            "This is usually a pyarrow/parquet compatibility issue rather than a plotting bug. "
            "Try upgrading/downgrading pyarrow on the machine where the dataset was generated."
        ) from exc
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    fps = float(info.get("fps", 0.0))

    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    layout = infer_layout(state_names, action_names)
    expected = recompute_expected_actions(states, layout)
    errors = np.abs(actions - expected)
    gripper_check = compare_gripper_alignment(states, actions, layout)

    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "debug_action_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"episode_{args.episode_index:06d}"
    save_action_plot(
        actions,
        expected,
        action_names,
        output_dir / f"{stem}_action_vs_recomputed.png",
        f"{stem}: stored vs recomputed action",
    )
    save_error_plot(
        errors,
        action_names,
        output_dir / f"{stem}_action_abs_error.png",
        f"{stem}: per-dimension absolute error",
    )
    save_norm_plot(
        actions,
        expected,
        output_dir / f"{stem}_action_norm.png",
        f"{stem}: action norm diagnostics",
    )

    summary = build_summary(actions, expected, action_names, fps, layout, gripper_check)
    summary_path = output_dir / f"{stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loaded: {parquet_path}")
    print(f"Frames: {len(states)}  FPS: {fps}")
    print(f"Overall MAE: {summary['overall_mae']:.8f}")
    print(f"Overall max abs error: {summary['overall_max_abs_error']:.8f}")
    if gripper_check is not None:
        print(f"Gripper alignment more likely matches: {gripper_check['more_likely_alignment']}")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved plots -> {output_dir}")


if __name__ == "__main__":
    main()
