#!/usr/bin/env python3
"""
Visualize ego-camera point clouds with hand trajectory overlay.

Renders each frame as a 3D scene (colored point cloud + wrist trajectory)
and saves to MP4, or opens an interactive window when a display is available.

Usage:
  # Save video (headless, works over SSH):
  python3 visualize_pointcloud_trajectory.py \
      --data_dir /home/ubuntu/orbbec/src/sync/collection_000/test/1 \
      --poses wrist_poses.npz \
      --output traj_pointcloud.mp4

  # Interactive window (requires display):
  python3 visualize_pointcloud_trajectory.py \
      --data_dir /home/ubuntu/orbbec/src/sync/collection_000/test/1 \
      --poses wrist_poses.npz \
      --interactive
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

DEPTH_SCALE = 0.001  # uint16 mm -> meters
Z_MIN, Z_MAX = 0.1, 3.0  # depth clip range (meters)
ARROW_LEN = 0.03  # orientation arrow length (meters)


# ── camera params ─────────────────────────────────────────────────────────────

def load_intrinsics(data_dir, cam_id):
    with open(Path(data_dir) / "camera_params.json") as f:
        p = json.load(f)
    cam = p.get(str(cam_id)) or p.get(cam_id)
    if cam is None:
        raise KeyError(f"Camera {cam_id} not found in camera_params.json")

    # Try multiple layouts:
    # 1. cam["Depth"]["intrinsic"]  (full format with separate depth section)
    # 2. cam["Depth"]               (depth section has fx/fy directly)
    # 3. cam["depth_intrinsic"]     (flat depth_intrinsic key)
    # 4. cam["rgb_intrinsic"]       (only RGB intrinsics available — use as fallback)
    # 5. cam["intrinsic"]           (single intrinsic block)
    def _try(d):
        if d and isinstance(d, dict) and "fx" in d:
            return d
        return None

    intr = (
        _try(cam.get("Depth", {}).get("intrinsic"))
        or _try(cam.get("Depth"))
        or _try(cam.get("depth_intrinsic"))
        or _try(cam.get("rgb_intrinsic"))
        or _try(cam.get("intrinsic"))
        or _try(cam)
    )
    if intr is None or "fx" not in intr:
        raise KeyError(f"Cannot parse intrinsics for camera {cam_id} from camera_params.json")

    result = {k: float(intr[k]) for k in ("fx", "fy", "cx", "cy")}
    result["width"] = float(intr.get("width", cam.get("Depth", {}).get("width", 848)))
    result["height"] = float(intr.get("height", cam.get("Depth", {}).get("height", 480)))
    return result


# ── pose loading ──────────────────────────────────────────────────────────────

def load_poses(poses_path):
    d = np.load(poses_path)
    return dict(
        left_pos=d["left_wrist_poses"][:, :3].astype(np.float32),
        left_rot=d["left_wrist_poses"][:, 3:].astype(np.float32),
        right_pos=d["right_wrist_poses"][:, :3].astype(np.float32),
        right_rot=d["right_wrist_poses"][:, 3:].astype(np.float32),
        left_valid=d["left_valid"].astype(bool),
        right_valid=d["right_valid"].astype(bool),
        fps=float(d["fps"]) if "fps" in d else 30.0,
    )


# ── point cloud ───────────────────────────────────────────────────────────────

def build_pointcloud(rgb_path, depth_path, intr, stride):
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    rgb_bgr = cv2.imread(str(rgb_path))
    if depth_raw is None or rgb_bgr is None:
        return None
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = depth_raw.astype(np.float32) * DEPTH_SCALE

    H, W = depth.shape
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]

    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    zs = depth[ys, xs]
    mask = (zs > Z_MIN) & (zs < Z_MAX)
    xs, ys, zs = xs[mask].ravel(), ys[mask].ravel(), zs[mask].ravel()

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    pts = np.stack([X, Y, zs], axis=1)
    colors = rgb[ys, xs].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ── trajectory geometry ───────────────────────────────────────────────────────

def build_trail(positions, valid, frame_idx, color):
    """LineSet of all frames up to frame_idx (full persistent trajectory)."""
    pts = []
    lines = []
    prev = None
    for i in range(frame_idx + 1):
        if not valid[i]:
            prev = None
            continue
        pts.append(positions[i])
        if prev is not None:
            lines.append([len(pts) - 2, len(pts) - 1])
        prev = i

    if len(pts) < 2:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def build_sphere(pos, radius, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(pos)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def build_orient_arrows(pos, euler_xyz, scale=ARROW_LEN):
    """Three small arrows (X=red, Y=green, Z=blue) at pos."""
    R_mat = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    meshes = []
    axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for axis_idx, col in enumerate(axis_colors):
        direction = R_mat[:, axis_idx]
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=scale * 0.08,
            cone_radius=scale * 0.15,
            cylinder_height=scale * 0.7,
            cone_height=scale * 0.3,
        )
        # Default arrow points along +Z; rotate to target direction
        z = np.array([0, 0, 1.0])
        v = np.cross(z, direction)
        s = np.linalg.norm(v)
        c = np.dot(z, direction)
        if s < 1e-6:
            if c > 0:
                R_align = np.eye(3)
            else:
                R_align = np.diag([1, -1, -1])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        T = np.eye(4)
        T[:3, :3] = R_align
        arrow.transform(T)
        arrow.translate(pos)
        arrow.paint_uniform_color(col)
        arrow.compute_vertex_normals()
        meshes.append(arrow)
    return meshes


# ── material helpers ──────────────────────────────────────────────────────────

def pcd_material():
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0
    return mat


def line_material(color):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 3.0
    mat.base_color = [*color, 1.0]
    return mat


def mesh_material():
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    return mat


# ── offscreen renderer ────────────────────────────────────────────────────────

def make_renderer(width, height):
    return o3d.visualization.rendering.OffscreenRenderer(width, height)


def setup_camera(renderer, width, height, intr, eye, lookat, up):
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    # Use intrinsic-based camera for accurate projection
    K = o3d.camera.PinholeCameraIntrinsic(
        int(width), int(height), fx, fy, cx, cy
    )
    extr = np.eye(4)
    # Build extrinsic from eye/lookat/up
    z = np.array(lookat) - np.array(eye)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array(up))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    extr[:3, 0] = x
    extr[:3, 1] = y
    extr[:3, 2] = z
    extr[:3, 3] = eye
    # open3d expects camera-to-world; invert for world-to-camera
    extr_inv = np.linalg.inv(extr)
    renderer.setup_camera(K, extr_inv)


def render_frame(renderer, pcd, geoms, width, height, intr, eye, lookat, up):
    renderer.scene.clear_geometry()
    renderer.scene.set_background([0.08, 0.08, 0.12, 1.0])

    # Lighting
    renderer.scene.scene.set_sun_light([0.5, -1, -0.5], [1, 1, 1], 50000)
    renderer.scene.scene.enable_sun_light(True)

    if pcd is not None and len(pcd.points) > 0:
        renderer.scene.add_geometry("pcd", pcd, pcd_material())

    for name, geom, mat in geoms:
        try:
            renderer.scene.add_geometry(name, geom, mat)
        except Exception:
            pass

    setup_camera(renderer, width, height, intr, eye, lookat, up)
    img = renderer.render_to_image()
    return np.asarray(img)


# ── interactive mode ──────────────────────────────────────────────────────────

def run_interactive(rgb_files, depth_files, poses, intr, stride, trail, fps):
    T = len(rgb_files)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Point Cloud Trajectory", width=int(intr["width"]), height=int(intr["height"]))

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    state = {"frame": 0, "paused": False}

    def next_frame(vis):
        state["frame"] = min(state["frame"] + 1, T - 1)

    def prev_frame(vis):
        state["frame"] = max(state["frame"] - 1, 0)

    def toggle_pause(vis):
        state["paused"] = not state["paused"]

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), prev_frame)
    vis.register_key_callback(ord(" "), toggle_pause)

    import time
    frame_dt = 1.0 / fps
    last_t = time.time()
    frame_idx = 0

    while vis.poll_events():
        now = time.time()
        if not state["paused"] and (now - last_t) >= frame_dt:
            frame_idx = (frame_idx + 1) % T
            last_t = now

        new_pcd = build_pointcloud(rgb_files[frame_idx], depth_files[frame_idx], intr, stride)
        if new_pcd is not None:
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            vis.update_geometry(pcd)

        vis.update_renderer()

    vis.destroy_window()


# ── video save mode ───────────────────────────────────────────────────────────

def run_save(rgb_files, depth_files, poses, intr, stride, trail, fps, output, width, height, eye, lookat, up):
    T = len(rgb_files)
    renderer = make_renderer(width, height)

    # Temp dir for PNG frames
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Rendering {T} frames...")

        for i in range(T):
            if i % 30 == 0:
                print(f"  frame {i}/{T}")

            pcd = build_pointcloud(rgb_files[i], depth_files[i], intr, stride)
            geoms = []

            # Left trail
            lt = build_trail(poses["left_pos"], poses["left_valid"], i, [0.2, 0.4, 1.0])
            if lt is not None:
                geoms.append(("left_trail", lt, line_material([0.2, 0.4, 1.0])))

            # Right trail
            rt = build_trail(poses["right_pos"], poses["right_valid"], i, [1.0, 0.3, 0.2])
            if rt is not None:
                geoms.append(("right_trail", rt, line_material([1.0, 0.3, 0.2])))

            # Current wrist spheres
            if poses["left_valid"][i]:
                s = build_sphere(poses["left_pos"][i], 0.015, [0.2, 0.5, 1.0])
                geoms.append(("left_sphere", s, mesh_material()))
                for j, arr in enumerate(build_orient_arrows(poses["left_pos"][i], poses["left_rot"][i])):
                    geoms.append((f"left_arrow_{j}", arr, mesh_material()))

            if poses["right_valid"][i]:
                s = build_sphere(poses["right_pos"][i], 0.015, [1.0, 0.4, 0.2])
                geoms.append(("right_sphere", s, mesh_material()))
                for j, arr in enumerate(build_orient_arrows(poses["right_pos"][i], poses["right_rot"][i])):
                    geoms.append((f"right_arrow_{j}", arr, mesh_material()))

            frame = render_frame(renderer, pcd, geoms, width, height, intr, eye, lookat, up)
            cv2.imwrite(str(tmpdir / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print("Encoding video with ffmpeg...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output),
        ]
        subprocess.run(cmd, check=True)
        print(f"Saved -> {output}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Point cloud + trajectory visualization")
    ap.add_argument("--data_dir", required=True, help="Sequence directory with camera_params.json and <cam_id>/RGB,Depth")
    ap.add_argument("--poses", default=None, help="wrist_poses.npz path (default: <data_dir>/wrist_poses.npz)")
    ap.add_argument("--output", default="traj_pointcloud.mp4", help="Output MP4 path")
    ap.add_argument("--cam_id", default="07", help="Camera folder ID (default: 07)")
    ap.add_argument("--stride", type=int, default=4, help="Depth downsample stride (default: 4)")
    ap.add_argument("--trail", type=int, default=30, help="Trajectory history frames (default: 30)")
    ap.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
    ap.add_argument("--width", type=int, default=None, help="Render width (default: from camera params)")
    ap.add_argument("--height", type=int, default=None, help="Render height (default: from camera params)")
    ap.add_argument("--interactive", action="store_true", help="Open interactive window (requires display)")
    ap.add_argument("--eye", type=float, nargs=3, default=[0.0, -0.3, -0.3],
                    metavar=("X", "Y", "Z"), help="Camera eye position (default: 0 -0.3 -0.3)")
    ap.add_argument("--lookat", type=float, nargs=3, default=[0.0, 0.0, 0.5],
                    metavar=("X", "Y", "Z"), help="Camera look-at point (default: 0 0 0.5)")
    ap.add_argument("--up", type=float, nargs=3, default=[0.0, -1.0, 0.0],
                    metavar=("X", "Y", "Z"), help="Camera up vector (default: 0 -1 0)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    poses_path = Path(args.poses) if args.poses else data_dir / "wrist_poses.npz"
    cam_dir = data_dir / args.cam_id

    rgb_dir = cam_dir / "RGB"
    depth_dir = cam_dir / "Depth"
    if not rgb_dir.exists():
        sys.exit(f"RGB directory not found: {rgb_dir}")
    if not depth_dir.exists():
        sys.exit(f"Depth directory not found: {depth_dir}")

    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    depth_files = sorted(depth_dir.glob("*.png"))
    if len(rgb_files) == 0:
        sys.exit(f"No JPG files in {rgb_dir}")
    if len(rgb_files) != len(depth_files):
        print(f"Warning: RGB ({len(rgb_files)}) and Depth ({len(depth_files)}) frame counts differ; using min")
        n = min(len(rgb_files), len(depth_files))
        rgb_files, depth_files = rgb_files[:n], depth_files[:n]

    intr = load_intrinsics(data_dir, args.cam_id)
    poses = load_poses(poses_path)

    T_data = len(poses["left_pos"])
    T_frames = len(rgb_files)
    if T_data != T_frames:
        print(f"Warning: poses ({T_data}) and frames ({T_frames}) differ; using min")
        n = min(T_data, T_frames)
        rgb_files = rgb_files[:n]
        depth_files = depth_files[:n]
        for k in poses:
            if isinstance(poses[k], np.ndarray) and poses[k].ndim >= 1 and len(poses[k]) == T_data:
                poses[k] = poses[k][:n]

    width = args.width or int(intr["width"])
    height = args.height or int(intr["height"])
    fps = poses["fps"] if poses["fps"] > 0 else args.fps

    print(f"Data dir:  {data_dir}")
    print(f"Frames:    {len(rgb_files)}  FPS: {fps:.1f}")
    print(f"Render:    {width}x{height}  stride={args.stride}  trail={args.trail}")
    print(f"Intrinsics: fx={intr['fx']:.2f} fy={intr['fy']:.2f} cx={intr['cx']:.2f} cy={intr['cy']:.2f}")

    if args.interactive:
        run_interactive(rgb_files, depth_files, poses, intr, args.stride, args.trail, fps)
    else:
        run_save(rgb_files, depth_files, poses, intr, args.stride, args.trail, fps,
                 args.output, width, height, args.eye, args.lookat, args.up)


if __name__ == "__main__":
    main()
