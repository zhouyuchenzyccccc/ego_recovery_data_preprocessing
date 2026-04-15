#!/usr/bin/env python3
"""
Visualize extracted hand trajectories in 3D space.

Supports:
  - .npz  (output of extract_wrist_pose.py)
  - .parquet  (LeRobot episode file)

Usage:
  python visualize_3d_trajectory.py wrist_poses.npz
  python visualize_3d_trajectory.py lerobot_dataset/data/chunk-000/episode_000000.parquet
  python visualize_3d_trajectory.py wrist_poses.npz --save traj.png
  python visualize_3d_trajectory.py wrist_poses.npz --stride 15 --no-orient
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation


# ── data loading ──────────────────────────────────────────────────────────────

def load_npz(path):
    d = np.load(path)
    left_pos   = d["left_wrist_poses"][:, :3].astype(np.float32)
    left_rot   = d["left_wrist_poses"][:, 3:].astype(np.float32)
    right_pos  = d["right_wrist_poses"][:, :3].astype(np.float32)
    right_rot  = d["right_wrist_poses"][:, 3:].astype(np.float32)
    left_grip  = d["left_gripper"].astype(np.float32)  if "left_gripper"  in d else np.zeros(len(left_pos))
    right_grip = d["right_gripper"].astype(np.float32) if "right_gripper" in d else np.zeros(len(right_pos))
    left_valid  = d["left_valid"].astype(bool)  if "left_valid"  in d else np.ones(len(left_pos),  dtype=bool)
    right_valid = d["right_valid"].astype(bool) if "right_valid" in d else np.ones(len(right_pos), dtype=bool)
    fps = float(d["fps"]) if "fps" in d else 30.0
    return dict(
        left_pos=left_pos, left_rot=left_rot, left_grip=left_grip, left_valid=left_valid,
        right_pos=right_pos, right_rot=right_rot, right_grip=right_grip, right_valid=right_valid,
        fps=fps,
    )


def load_parquet(path):
    import pyarrow.parquet as pq
    # Use iter_batches to avoid pyarrow histogram-mismatch bug with some parquet files
    f = pq.ParquetFile(path)
    rows = []
    for batch in f.iter_batches(columns=["observation.state"]):
        rows.extend(batch.column("observation.state").to_pylist())
    state = np.array(rows, dtype=np.float32)

    # Try to detect layout from info.json next to the dataset root
    # Walk up to find meta/info.json
    info_path = None
    for parent in Path(path).parents:
        candidate = parent / "meta" / "info.json"
        if candidate.exists():
            info_path = candidate
            break

    has_gripper = False
    fps = 30.0
    if info_path is not None:
        import json
        with open(info_path) as f_info:
            info = json.load(f_info)
        fps = float(info.get("fps", 30.0))
        names = info.get("features", {}).get("observation.state", {}).get("names", [])
        has_gripper = "left_gripper" in names or "right_gripper" in names

    T = len(state)
    if has_gripper:
        # 14-D: [lx,ly,lz,lrx,lry,lrz,lg, rx,ry,rz,rrx,rry,rrz,rg]
        left_pos  = state[:, 0:3]
        left_rot  = state[:, 3:6]
        left_grip = state[:, 6]
        right_pos  = state[:, 7:10]
        right_rot  = state[:, 10:13]
        right_grip = state[:, 13]
    else:
        # 12-D: [lx,ly,lz,lrx,lry,lrz, rx,ry,rz,rrx,rry,rrz]
        left_pos  = state[:, 0:3]
        left_rot  = state[:, 3:6]
        left_grip = np.zeros(T, dtype=np.float32)
        right_pos  = state[:, 6:9]
        right_rot  = state[:, 9:12]
        right_grip = np.zeros(T, dtype=np.float32)

    return dict(
        left_pos=left_pos, left_rot=left_rot, left_grip=left_grip,
        left_valid=np.ones(T, dtype=bool),
        right_pos=right_pos, right_rot=right_rot, right_grip=right_grip,
        right_valid=np.ones(T, dtype=bool),
        fps=fps,
    )


def load_data(path):
    p = Path(path)
    if p.suffix == ".npz":
        return load_npz(p)
    if p.suffix == ".parquet":
        return load_parquet(p)
    raise ValueError(f"Unsupported file type: {p.suffix}  (expected .npz or .parquet)")


# ── helpers ───────────────────────────────────────────────────────────────────

def colored_segments(ax, xyz, colors):
    """Draw a 3D polyline with per-segment colors."""
    pts = xyz.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, colors=colors, linewidth=1.5, alpha=0.85)
    ax.add_collection3d(lc)


def draw_orient_arrows(ax, pos, rot_euler, valid, stride, scale):
    """Draw small RGB orientation axes at sampled frames."""
    indices = range(0, len(pos), stride)
    for i in indices:
        if not valid[i]:
            continue
        R_mat = Rotation.from_euler("xyz", rot_euler[i]).as_matrix()
        o = pos[i]
        for col, axis_idx in [("red", 0), ("green", 1), ("blue", 2)]:
            d = R_mat[:, axis_idx] * scale
            ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                      color=col, linewidth=0.8, arrow_length_ratio=0.3)


def gripper_sizes(grip, s_min=10, s_max=60):
    """Map gripper [0,1] → marker size (open=large, closed=small)."""
    g = np.clip(grip, 0.0, 1.0)
    return s_min + g * (s_max - s_min)


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_trajectories(data, stride, show_orient, save_path):
    left_pos   = data["left_pos"]
    left_rot   = data["left_rot"]
    left_grip  = data["left_grip"]
    left_valid = data["left_valid"]
    right_pos  = data["right_pos"]
    right_rot  = data["right_rot"]
    right_grip = data["right_grip"]
    right_valid = data["right_valid"]
    T = len(left_pos)

    t = np.linspace(0, 1, T)
    cmap_l = plt.cm.Blues
    cmap_r = plt.cm.Reds

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ── trajectories (color = time) ──
    seg_colors_l = cmap_l(t[:-1] * 0.6 + 0.3)
    seg_colors_r = cmap_r(t[:-1] * 0.6 + 0.3)
    colored_segments(ax, left_pos,  seg_colors_l)
    colored_segments(ax, right_pos, seg_colors_r)

    # ── scatter (gripper size) ──
    sz_l = gripper_sizes(left_grip)
    sz_r = gripper_sizes(right_grip)
    sc_l = ax.scatter(left_pos[:, 0],  left_pos[:, 1],  left_pos[:, 2],
                      c=t, cmap="Blues", s=sz_l, alpha=0.6, vmin=0, vmax=1, label="Left hand")
    sc_r = ax.scatter(right_pos[:, 0], right_pos[:, 1], right_pos[:, 2],
                      c=t, cmap="Reds",  s=sz_r, alpha=0.6, vmin=0, vmax=1, label="Right hand")

    # ── mark invalid (interpolated) frames ──
    li = ~left_valid
    ri = ~right_valid
    if li.any():
        ax.scatter(left_pos[li, 0], left_pos[li, 1], left_pos[li, 2],
                   marker="x", s=20, color="cornflowerblue", alpha=0.5, label="Left (interp)")
    if ri.any():
        ax.scatter(right_pos[ri, 0], right_pos[ri, 1], right_pos[ri, 2],
                   marker="x", s=20, color="salmon", alpha=0.5, label="Right (interp)")

    # ── start / end markers ──
    for pos, col in [(left_pos, "blue"), (right_pos, "red")]:
        ax.scatter(*pos[0],  marker="o", s=80, color=col, edgecolors="white", zorder=5)
        ax.scatter(*pos[-1], marker="*", s=120, color=col, edgecolors="white", zorder=5)

    # ── orientation arrows ──
    if show_orient:
        all_pos = np.concatenate([left_pos, right_pos], axis=0)
        span = np.ptp(all_pos, axis=0).max()
        arrow_scale = span * 0.04
        draw_orient_arrows(ax, left_pos,  left_rot,  left_valid,  stride, arrow_scale)
        draw_orient_arrows(ax, right_pos, right_rot, right_valid, stride, arrow_scale)

    # ── colorbars ──
    sm_l = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 1))
    sm_r = plt.cm.ScalarMappable(cmap="Reds",  norm=plt.Normalize(0, 1))
    sm_l.set_array([])
    sm_r.set_array([])
    cb_l = fig.colorbar(sm_l, ax=ax, shrink=0.4, pad=0.02, location="left")
    cb_r = fig.colorbar(sm_r, ax=ax, shrink=0.4, pad=0.02, location="right")
    cb_l.set_label("Left — time", fontsize=9)
    cb_r.set_label("Right — time", fontsize=9)

    # ── axes labels ──
    ax.set_xlabel("X  (right, m)")
    ax.set_ylabel("Y  (down, m)")
    ax.set_zlabel("Z  (forward, m)")
    ax.set_title(f"Hand Trajectories — {T} frames  |  ○=start  ★=end  ×=interpolated\n"
                 f"Marker size ∝ gripper openness", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)

    # equal aspect ratio
    all_pos = np.concatenate([left_pos, right_pos], axis=0)
    mn, mx = all_pos.min(axis=0), all_pos.max(axis=0)
    mid = (mn + mx) / 2
    half = (mx - mn).max() / 2 * 1.1
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Visualize hand trajectories in 3D")
    ap.add_argument("input", nargs="?", default="wrist_poses.npz",
                    help=".npz or .parquet file (default: wrist_poses.npz)")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help="Save figure to PNG instead of showing interactively")
    ap.add_argument("--stride", type=int, default=10,
                    help="Sample every N frames for orientation arrows (default: 10)")
    ap.add_argument("--no-orient", action="store_true",
                    help="Skip orientation arrow rendering")
    args = ap.parse_args()

    print(f"Loading: {args.input}")
    data = load_data(args.input)
    T = len(data["left_pos"])
    fps = data["fps"]
    print(f"Frames: {T}  FPS: {fps:.1f}  Duration: {T/fps:.1f}s")
    print(f"Left  valid: {data['left_valid'].sum()}/{T} ({100*data['left_valid'].mean():.1f}%)")
    print(f"Right valid: {data['right_valid'].sum()}/{T} ({100*data['right_valid'].mean():.1f}%)")

    plot_trajectories(data, stride=args.stride, show_orient=not args.no_orient, save_path=args.save)


if __name__ == "__main__":
    main()
