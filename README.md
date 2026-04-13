# EgoRecovery 数据预处理工具

## 📖 项目简介

本项目用于将 Ego2（第一人称视角）数据转换为 LeRobot v2 数据集格式，主要用于人体手部恢复（ego recovery）示范数据的预处理。项目包含三个核心脚本：

1. **extract_wrist_pose.py** - 从 RGB-D 数据中提取 3D 手腕位姿
2. **visualize_wrist_pose.py** - 可视化提取的手腕位姿结果
3. **convert_to_lerobot.py** - 将处理后的数据转换为 LeRobot 数据集格式

## 🛠️ 环境依赖

```bash
pip install numpy opencv-python scipy mediapipe pandas pyarrow
```

还需要安装 `ffmpeg` 用于视频编码：
```bash
sudo apt install ffmpeg
```

## 📋 数据目录结构

输入数据应遵循以下目录结构：

```
<data_dir>/
├── camera_params.json          # 相机内参文件
├── 07/                         # 第一人称相机（ego camera）
│   ├── RGB/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   └── Depth/
│       ├── 00000.png
│       ├── 00001.png
│       └── ...
├── 06/                         # 左手相机
│   └── RGB/
└── 08/                         # 右手相机
    └── RGB/
```

## 🚀 使用步骤

### 步骤 1：提取手腕位姿

从 RGB-D 图像中提取 3D 手腕位姿（位置和欧拉角）：

```bash
python3 extract_wrist_pose.py \
    --data_dir /path/to/ego2_data \
    --output wrist_poses.npz \
    --model_path /path/to/hand_landmarker.task \
    --cam_id 07 \
    --smooth_method median_then_savgol
```

**主要参数：**
- `--data_dir`: 输入数据目录
- `--output`: 输出的位姿文件路径（.npz 格式）
- `--model_path`: MediaPipe 手部关键点检测模型路径
- `--cam_id`: 相机 ID（默认为 "07"）
- `--smooth_method`: 平滑方法，可选 `savgol`、`ema`、`median_then_savgol`、`none`

**输出内容：**
- `left_wrist_poses`: 左手位姿数组 (T, 6)，包含 [x, y, z, rx, ry, rz]
- `right_wrist_poses`: 右手位姿数组 (T, 6)
- `left_valid`: 左手检测有效帧标记
- `right_valid`: 右手检测有效帧标记
- `fps`: 帧率（默认 30.0）

### 步骤 2：可视化验证

验证提取的手腕位姿是否准确：

```bash
python3 visualize_wrist_pose.py \
    --data_dir /path/to/ego2_data \
    --poses wrist_poses.npz \
    --output wrist_viz.mp4 \
    --model_path /path/to/hand_landmarker.task \
    --cam_id 07 \
    --fps 30.0
```

**主要参数：**
- `--data_dir`: 输入数据目录
- `--poses`: 步骤 1 生成的位姿文件
- `--output`: 输出的可视化视频路径
- `--max_frames`: 最大处理帧数（可选，用于快速测试）

**可视化内容：**
- MediaPipe 手部关键点骨架
- 3D 坐标轴（X=红色, Y=绿色, Z=蓝色）
- 位姿信息文本（位置和检测状态）

### 步骤 3：转换为 LeRobot 数据集

将处理后的数据转换为 LeRobot v2 格式：

```bash
python3 convert_to_lerobot.py \
    --data_dir /path/to/ego2_data \
    --poses wrist_poses.npz \
    --output lerobot_dataset \
    --task "human ego recovery demonstration" \
    --fps 30.0
```

**主要参数：**
- `--data_dir`: 输入数据目录
- `--poses`: 步骤 1 生成的位姿文件
- `--output`: 输出的 LeRobot 数据集目录
- `--task`: 任务描述
- `--fps`: 帧率（默认 30.0）
- `--episode_index`: 片段索引（多片段时递增，默认为 0）
- `--no_video`: 跳过视频编码（用于快速测试）

**输出目录结构：**

```
<output>/
├── meta/
│   ├── info.json              # 数据集元信息
│   ├── episodes.jsonl         # 片段信息
│   ├── tasks.jsonl            # 任务信息
│   └── stats.json             # 统计数据
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet  # 位姿和动作数据
└── videos/
    └── chunk-000/
        ├── observation.images.ego/
        │   └── episode_000000.mp4
        ├── observation.images.left_hand/
        │   └── episode_000000.mp4
        └── observation.images.right_hand/
            └── episode_000000.mp4
```

## 📊 数据格式说明

### 状态（State）

12 维向量，包含左右手腕的 6D 位姿：
- `left_wrist_[x, y, z, rx, ry, rz]`
- `right_wrist_[x, y, z, rx, ry, rz]`

### 动作（Action）

12 维向量，包含左右手腕的位姿增量（delta）：
- `left_delta_[x, y, z, rx, ry, rz]`
- `right_delta_[x, y, z, rx, ry, rz]`

角度增量使用 `angle_wrap` 进行 [-π, π]  wrap。

### 视频（Videos）

三个相机视角的 H.264 编码 MP4 视频：
- `observation.images.ego`: 第一人称视角（相机 07）
- `observation.images.left_hand`: 左手视角（相机 06）
- `observation.images.right_hand`: 右手视角（相机 08）

## 🔧 多片段数据处理

当有多个片段需要处理时，重复运行转换脚本并递增 `--episode_index`：

```bash
# 第一个片段
python3 convert_to_lerobot.py --data_dir /path/to/ego2_1 --poses poses_1.npz --output lerobot_dataset --episode_index 0

# 第二个片段
python3 convert_to_lerobot.py --data_dir /path/to/ego2_2 --poses poses_2.npz --output lerobot_dataset --episode_index 1
```

脚本会自动累积元数据（info.json、episodes.jsonl 等）。

## 📝 注意事项

1. **深度数据缩放因子**: 代码中 `DEPTH_SCALE = 0.001` 适用于 Orbbec 相机（uint16 mm → 米）。如果使用其他深度相机，请调整此值。
2. **MediaPipe 模型方向**: MediaPipe 假设前置/自拍视角，代码中已自动翻转左右手检测以适应固定第一人称相机。
3. **无效帧处理**: 未检测到手部的帧会使用插值填充，并在可视化中标记为 "INTERP"。
4. **平滑方法**: 推荐使用 `median_then_savgol`（先中值滤波再 Savgol 滤波）以获得最佳平滑效果。

## 📄 许可证

[请根据实际情况添加许可证信息]

## 🤝 贡献

[请根据实际情况添加贡献指南]

## Download MediaPipe Model

Download the `hand_landmarker.task` model to the default path used by `extract_wrist_pose.py`:

```bash
mkdir -p /home/ubuntu/WorkSpace/ZYC/hamer/_DATA/mediapipe && \
wget -O /home/ubuntu/WorkSpace/ZYC/hamer/_DATA/mediapipe/hand_landmarker.task \
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

After downloading, you can run:

```bash
python3 extract_wrist_pose.py \
    --data_dir /path/to/ego2_data \
    --output wrist_poses.npz \
    --cam_id 07 \
    --smooth_method median_then_savgol
```

## Explicit Camera Mapping

If camera ids 06 / 07 / 08 may be swapped, always pass the camera mapping explicitly.

### Extract wrist pose from the ego camera

```bash
python3 extract_wrist_pose.py \
    --data_dir /path/to/ego2_data \
    --output wrist_poses.npz \
    --ego_cam_id 07 \
    --smooth_method median_then_savgol
```

### Convert to LeRobot with explicit video-camera mapping

```bash
python3 convert_to_lerobot.py \
    --data_dir /path/to/ego2_data \
    --output lerobot_dataset \
    --ego_cam_id 07 \
    --left_wrist_cam_id 06 \
    --right_wrist_cam_id 08 \
    --task "hand success insertion"
```

If your hardware wiring is swapped, just change the three ids above instead of modifying the code.

Note: in batch mode, if you pass `--output wrist_poses.npz`, the script will save files like `0_wrist_poses.npz`, `1_wrist_poses.npz`, ... in the current directory instead of creating a directory named `wrist_poses.npz`.
