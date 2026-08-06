# RoboClean - LeRobot数据清洗与可视化工具

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

通用的LeRobot数据集清洗和可视化工具，支持多种LeRobot版本（v2.0、v2.1、v3.0）的数据处理。

## 功能特性

- **多版本支持**: 自动识别并处理v2.0、v2.1、v3.0版本的LeRobot数据集
- **智能清洗**: 基于运动阈值的静态帧去除，零维度过滤
- **可视化分析**: 多种数据可视化工具，包括运动轨迹、帧分布、热力图等
- **Web界面**: 直观的Web界面浏览和播放数据集中的视频
- **通用设计**: 支持不同的机器人本体、任务类型和数据结构
- **易用命令行**: 简洁的命令行接口，支持多种操作

## v3.0支持亮点

RoboClean完全支持LeRobot v3.0数据格式，包括：

- **多视频文件处理**: 自动处理因文件大小限制而分割的多个视频文件（如file-000.mp4, file-001.mp4）
- **精确时间戳**: 从episodes元数据中读取精确的视频时间戳范围
- **共享数据文件**: 正确处理多个episode共享单个parquet文件的数据结构
- **智能列识别**: 自动识别v3.0的observation.state和action列

## 项目结构

```
RoboClean/
├── roboclean/              # 主包
│   ├── core/               # 核心功能
│   │   ├── loader.py       # 数据加载器（支持v2.x和v3.0）
│   │   └── cleaner.py      # 数据清洗器
│   ├── viz/                # 可视化
│   │   └── visualizer.py   # 可视化工具
│   ├── web/                # Web界面
│   │   └── app.py          # Flask应用
│   └── cli/                # 命令行工具
│       └── main.py         # CLI主程序
├── demo_data/              # 示例数据
├── requirements.txt        # 依赖包
├── setup.py                # 安装脚本
└── README.md               # 本文件
```

## 环境搭建

### 系统要求

- Python 3.8 或更高版本
- 建议使用conda或venv虚拟环境

### 快速安装

```bash
# 创建conda环境（推荐）
conda create -n roboclean python=3.9 -y
conda activate roboclean

# 克隆项目（如果还没有）
cd /path/to/your/workspace
git clone <repository_url>
cd RoboClean

# 安装依赖和项目包（必须安装才能使用roboclean命令）
pip install -r requirements.txt
pip install -e .

# 验证安装
roboclean --help
```

### 核心依赖

- numpy >= 1.21.0
- pyarrow >= 10.0.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- tqdm >= 4.62.0
- Flask >= 2.0.0
- rich >= 10.0.0

可选依赖（用于视频处理）：
- opencv-python >= 4.5.0
- av >= 8.0.0

## 使用方法

### 1. 命令行工具

#### 数据清洗

```bash
# 基础清洗
roboclean clean <dataset_path>

# 指定输出路径
roboclean clean <dataset_path> --output-path <output_path>

# 设置运动阈值
roboclean clean <dataset_path> --threshold 0.01

# 原地清洗（会备份原数据）
roboclean clean <dataset_path> --in-place

# 查看帮助
roboclean clean --help
```

#### 可视化分析

```bash
# 生成可视化报告
roboclean visualize <dataset_path> --output-dir <output_dir>

# 显示episode长度分布
roboclean visualize <dataset_path> --show-lengths
```

### 2. Web界面

```bash
# 启动Web界面
roboclean web

# 指定端口
roboclean web --port 8080

# 启动时加载数据集
roboclean web --dataset-path <dataset_path>
```

然后在浏览器中打开 `http://localhost:5000` 即可查看数据集。

### 3. Python API

```python
from roboclean.core.loader import LeRobotDatasetLoader
from roboclean.core.cleaner import DataCleaner
from roboclean.viz.visualizer import DatasetVisualizer

# 加载数据集（自动识别版本）
loader = LeRobotDatasetLoader('/path/to/dataset')
print(f"版本: {loader.version}")
print(f"Episodes: {len(loader.episodes)}")

# 数据清洗
cleaner = DataCleaner(loader)
cleaner.clean_dataset(
    output_root='/path/to/output',
    threshold=1e-3
)

# 可视化
viz = DatasetVisualizer(loader)
viz.print_episode_lengths()
fig = viz.plot_episode_length_distribution()
fig.savefig('episode_lengths.png')
```

## 数据清洗参数说明

### 核心参数

#### `--threshold`
- **默认值**: 1e-3
- **说明**: 运动阈值，小于此值的帧将被视为静止帧并移除
- **建议**:
  - 小阈值（1e-4 - 1e-3）: 保留更多细节，适合精细操作任务
  - 中阈值（1e-3 - 1e-2）: 平衡去噪和保留，适合一般任务
  - 大阈值（1e-2 - 1e-1）: 强力去噪，适合高速运动任务

#### `--norm`
- **默认值**: max
- **选项**: max, l2
- **说明**: 运动范数计算方法
- **max**: 使用各维度最大差值（推荐，更保守）
- **l2**: 使用欧几里得范数（对整体运动更敏感）

#### `--compare-to`
- **默认值**: previous
- **选项**: previous, last-kept
- **说明**: 帧对比方法
- **previous**: 与前一帧对比（检测连续运动）
- **last-kept**: 与上一个保留的帧对比（检测累积运动）

### 高级参数

#### `--drop-zero-rows`
移除所有运动列全为0的帧（可能是记录错误）

#### `--zero-dim-specs`
指定特定维度必须为0的行将被移除
```bash
--zero-dim-specs '{"action": [5]}'  # 移除action第5维为0的帧
```

#### `--drop-empty-episodes`
清洗后如果episode为空，则完全移除该episode

## 数据清洗逻辑

### 静止帧检测

RoboClean通过以下步骤检测静止帧：

1. **计算帧间运动**: 计算当前帧与参考帧（前一帧或上一保留帧）的差异
2. **运动范数**: 使用指定范数（max或l2）计算运动强度
3. **阈值判断**: 运动强度低于阈值的帧被视为静止帧
4. **保留策略**: 始终保留第一帧，避免episode为空

### 清洗流程

```
原始数据
    ↓
加载episode数据
    ↓
计算每帧运动分数
    ↓
应用运动阈值过滤
    ↓
重新索引（frame_index, timestamp, index）
    ↓
保存清洗后的数据
    ↓
更新元数据（episodes.jsonl, info.json）
```

## v3.0数据处理示例

### 查看v3.0数据集信息

```python
from roboclean.core.loader import LeRobotDatasetLoader

loader = LeRobotDatasetLoader('demo_data/pickbread/00')
print(f"版本: {loader.version}")  # v3.0
print(f"视频列: {loader.get_image_columns()}")

# 查看episode的视频信息
video_info = loader.get_episode_video_info(0)
for camera, info in video_info.items():
    print(f"{camera}:")
    print(f"  文件: {info['path']}")
    print(f"  时间: {info['from_timestamp']:.3f}s - {info['to_timestamp']:.3f}s")
```

### 清洗v3.0数据

```python
from roboclean.core.cleaner import DataCleaner

cleaner = DataCleaner(loader)
output_path = cleaner.clean_dataset(
    output_root='demo_data/pickbread/00_cleaned',
    threshold=1e-3
)
print(f"清洗完成: {output_path}")
```

## 实际案例

### 案例0: 清洗v3.0数据集（推荐新手）

**数据集信息：**
- 路径：`/data/chenxugao/RoboClean/demo_data/pickbread/00`
- 版本：v3.0
- Episodes：30个
- 总帧数：13879帧
- 机器人：so_follower

**命令行清洗：**

```bash
# 激活roboclean环境（必须！）
conda activate roboclean

# 基础清洗（保留约75%的帧）
roboclean clean /data/chenxugao/RoboClean/demo_data/pickbread/00 \
    --output-path /data/chenxugao/RoboClean/demo_data/pickbread/00_cleaned

# 完整参数清洗示例
roboclean clean /data/chenxugao/RoboClean/demo_data/pickbread/00 \
    --output-path /data/chenxugao/RoboClean/demo_data/pickbread/00_cleaned \
    --threshold 1e-3 \
    --norm max \
    --compare-to previous \
    --drop-zero-rows \
    --drop-empty-episodes

# 参数说明：
# --threshold 1e-3         运动阈值（推荐1e-3到1e-2）
# --norm max               运动范数：max（推荐）或 l2
# --compare-to previous    对比方式：previous（推荐）或 last-kept
# --drop-zero-rows         移除所有运动为0的帧
# --drop-empty-episodes    移除清洗后为空的episode

# 实际运行结果示例：
# Episode 0: 485 -> 368 frames
# Episode 1: 452 -> 350 frames
# ...
# Total: 13879 -> 10709 frames (dropped 3170, 保留率77.1%)
```

**Python API清洗（更灵活）：**

```python
from roboclean.core.loader import LeRobotDatasetLoader
from roboclean.core.cleaner import DataCleaner

# 1. 加载数据集
loader = LeRobotDatasetLoader('/data/chenxugao/RoboClean/demo_data/pickbread/00')
print(f"加载数据集: {len(loader.episodes)} episodes, 版本: {loader.version}")

# 2. 创建清洗器
cleaner = DataCleaner(loader)

# 3. 执行清洗（使用默认参数）
output_path = cleaner.clean_dataset(
    output_root='/data/chenxugao/RoboClean/demo_data/pickbread/00_cleaned',
    threshold=1e-3,  # 运动阈值
    norm='max',       # 使用max范数
    compare_to='previous'  # 与前一帧对比
)

print(f"清洗完成！输出路径: {output_path}")

# 4. 验证清洗结果
cleaned_loader = LeRobotDatasetLoader(output_path)
print(f"清洗后: {len(cleaned_loader.episodes)} episodes")
print(f"清洗后总帧数: {cleaned_loader.get_dataset_stats()['total_frames']}")
```

**预期效果：**
- ✅ 移除静止帧，保留运动帧
- ✅ 数据量减少约20-30%
- ✅ 训练效率提升，模型更关注有意义的数据
- ✅ 自动处理v3.0的多视频文件格式

### 案例1: 去除机械臂保持静止的帧

```bash
roboclean clean demo_data/pick_from_shelf_0723 \
    --threshold 1e-3 \
    --norm max \
    --output-path cleaned_data
```

**效果**: 保留了所有运动帧，移除了机械臂保持静止的帧，数据量减少约25%

### 案例2: 过滤夹爪完全关闭的帧

```bash
roboclean clean demo_data/data_008 \
    --threshold 1e-3 \
    --zero-dim-specs '{"action": [5]}' \
    --drop-zero-rows
```

**效果**: 移除了夹爪完全关闭（action[5]=0）的帧和所有运动为0的帧

### 案例3: 强力去噪（高速运动）

```bash
roboclean clean demo_data/fast_motion \
    --threshold 1e-2 \
    --norm l2 \
    --compare-to last-kept
```

**效果**: 只保留显著运动的帧，适合高速抓取任务

## Web界面功能

RoboClean的Web界面提供：

- 📊 数据集概览（版本、episode数量、总帧数）
- 🎬 视频播放（支持v3.0多视频文件）
- 📏 Episode长度统计
- 📈 运动分析图表
- 🖼️ 多相机视角同步播放

启动Web界面：
```bash
roboclean web --dataset-path demo_data/pickbread/00
```

## 常见问题

### Q: 支持哪些LeRobot版本？
A: 支持v2.0、v2.1和v3.0，自动识别版本并适配处理方式。

### Q: v3.0的视频文件被分成了多个怎么办？
A: RoboClean会自动从episodes元数据中读取正确的视频文件和时间戳，无需手动处理。

### Q: 如何选择合适的threshold？
A: 建议从小值（1e-3）开始，根据保留的帧比例逐步调整：
- 保留比例>80%: 可以增大阈值
- 保留比例<50%: 可能需要减小阈值

### Q: 清洗会修改原始数据吗？
A: 默认不会，会创建新的清洗后数据集。使用`--in-place`会备份后修改原数据。

### Q: 如何处理清洗后的空episode？
A: 使用`--drop-empty-episodes`参数完全移除空episode，或者调整阈值保留至少一帧。

## 开发与测试

### 运行测试

```bash
# 安装开发依赖
pip install pytest pytest-cov

# 运行测试
pytest tests/

# 生成覆盖率报告
pytest --cov=roboclean tests/
```

### 代码风格

```bash
# 格式化代码
pip install black isort
black roboclean/
isort roboclean/
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0 (2025-01-XX)
- ✨ 完整支持LeRobot v3.0格式
- 🎬 支持多视频文件自动处理
- 🌐 新增Web可视化界面
- 📊 增强可视化分析功能
- 🐛 修复v3.0数据加载bug

## 致谢

感谢LeRobot项目提供的优秀数据格式和工具链。