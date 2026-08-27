在 MAPTR 项目中同时使用 nuPlan 和 nuScenes 数据集，通常需要将 nuPlan 数据转换为 nuScenes 的格式，以便复用 MAPTR 现有的数据加载、预处理和训练流程。以下是一个保姆级教程，涵盖从环境准备到训练验证的完整步骤。

---

## 1. 理解背景

- **MAPTR**：一种基于向量化地图的轨迹预测模型，原生支持 nuScenes 数据集。其数据加载器期望 nuScenes 的标准目录结构和标注格式（如 `sample_annotation.json`、`sample.json`、`scene.json` 等）。
- **nuScenes**：提供丰富的传感器数据和轨迹标注，以“场景（scene）”为单位组织，每个场景包含多个样本（sample），每个样本有对应的车辆状态和未来轨迹。
- **nuPlan**：另一个大规模自动驾驶数据集，包含更多场景和更复杂的驾驶行为。其数据组织方式与 nuScenes 不同，使用自己的数据库格式（`.db` 文件）和 API。

因此，对齐的核心工作是将 nuPlan 的数据转换成 nuScenes 的格式，然后将两个数据集合并或混合训练。

---

## 2. 整体思路

1. 使用 nuPlan-devkit 读取 nuPlan 数据，提取必要信息（自车轨迹、地图、其他智能体等）。
2. 按照 nuScenes 的 JSON 格式生成对应的元数据文件（`scene.json`、`sample.json`、`sample_annotation.json`、`ego_pose.json`、`instance.json` 等）。
3. 将 nuPlan 的地图数据转换为 nuScenes 的地图格式（通常是二进制 `.pkl` 文件或矢量地图）。
4. 将转换后的数据放在 nuScenes 目录结构下，或与原始 nuScenes 合并。
5. 修改 MAPTR 的配置，使其能够加载合并后的数据集。

---

## 3. 环境准备

```bash
# 创建虚拟环境（可选）
conda create -n maptr_nuplan python=3.8
conda activate maptr_nuplan

# 安装 MAPTR 依赖
git clone https://github.com/hustvl/MapTR.git
cd MapTR
pip install -r requirements.txt

# 安装 nuPlan devkit（用于读取 nuPlan 数据）
pip install nuplan-devkit
```

---

## 4. 下载并准备数据

### 4.1 下载 nuScenes 数据集（如已有可跳过）
从 [nuScenes 官网](https://www.nuscenes.org/) 下载完整数据集（包括 trainval 和 test），解压后得到标准目录结构：
```
nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
│   ├── scene.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── category.json
│   └── ...
```

### 4.2 下载 nuPlan 数据集
从 [nuPlan 官网](https://www.nuscenes.org/nuplan) 下载 nuPlan 数据集（例如 mini 版本用于测试），解压后得到 `.db` 文件和地图文件。通常结构为：
```
nuplan/
├── maps/
│   ├── us-nv-las-vegas-*.pdb
│   └── ...
└── nuplan-v1.1/
    ├── mini/
    │   ├── *.db
    │   └── ...
```

---

## 5. 理解 nuScenes 数据格式

在转换前，必须熟悉 nuScenes 的 JSON 文件结构：

- **scene.json**：每个场景的元数据，包括场景 ID、名称、描述、样本数量等。
- **sample.json**：每个样本（时间点）的信息，包括时间戳、自车位姿 token、场景 token 等。
- **sample_annotation.json**：每个样本中所有目标（车辆、行人等）的标注，包括目标 ID、类别、3D 框、属性等。
- **ego_pose.json**：自车在每个样本中的位姿（全局坐标系）。
- **instance.json**：跨样本的实例（同一目标在不同样本中的标识）。
- **category.json**：类别定义（如 vehicle、pedestrian 等）。

这些 JSON 文件通过 token 相互关联，形成完整的数据图。

---

## 6. 转换 nuPlan 数据为 nuScenes 格式

由于 nuPlan 与 nuScenes 的组织方式差异较大，转换脚本需要完成以下映射：

| nuPlan 概念 | nuScenes 对应 |
|-------------|---------------|
| 场景（Scenario） | scene |
| 时间帧（Frame） | sample |
| 自车状态（Ego pose） | ego_pose |
| 检测到的目标（Tracked object） | sample_annotation + instance |
| 地图（Map） | map（二进制或矢量） |
| 未来轨迹（可选） | 不直接在 nuScenes 标注中，MAPTR 会从样本序列中提取未来轨迹 |

### 6.1 提取 nuPlan 数据

使用 nuPlan-devkit 的 API 遍历场景和帧，提取所需信息。以下是一个简化的代码框架：

```python
from nuplan.database.nuplan_db import NuPlanDB
from nuplan.database.nuplan_db_utils import get_scenario_tokens
import os

db_path = "path/to/nuplan/mini/*.db"
map_path = "path/to/nuplan/maps/us-nv-las-vegas-*.pdb"

# 打开数据库
db = NuPlanDB(
    data_path=db_path,
    map_path=map_path,
    verbose=False
)

# 获取所有场景 token
scenario_tokens = get_scenario_tokens(db)
for scenario_token in scenario_tokens:
    scenario = db.scenario.get(scenario_token)
    # 场景信息：场景 ID、时间范围等
    # 获取场景中的所有帧（lidar_pc_tokens 或 ego_pose_tokens）
    frames = db.lidar_pc  # 或 db.ego_pose，取决于如何定义帧
    for frame in frames:
        # 提取自车位姿
        ego_pose = frame.ego_pose
        # 提取该帧中的所有目标（tracked objects）
        objects = frame.objects  # 或通过 db.track 获取
        for obj in objects:
            # 目标类别、3D框、轨迹等
            pass
```

### 6.2 生成 nuScenes JSON 文件

你需要创建以下 JSON 文件（格式严格遵循 nuScenes 规范）：

- **category.json**：可直接从 nuScenes 复制，或映射 nuPlan 的类别到 nuScenes 类别。
- **instance.json**：为每个持续跟踪的目标分配一个 instance token，跨帧保持一致。
- **scene.json**：每个 nuPlan 场景对应一个 scene 条目。
- **sample.json**：每个 nuPlan 帧对应一个 sample，记录时间戳、自车位姿 token、场景 token。
- **ego_pose.json**：每个自车位姿一个条目，包含全局坐标和旋转。
- **sample_annotation.json**：每个目标在每个帧中的标注，包含实例 token、类别 token、3D框（中心、尺寸、旋转）等。

以下是一个生成这些 JSON 的示例代码片段（伪代码，需根据实际数据结构调整）：

```python
import json
from pyquaternion import Quaternion

categories = [
    {"token": "vehicle", "name": "vehicle", "description": "Vehicle"},
    {"token": "pedestrian", "name": "pedestrian", "description": "Pedestrian"},
    # 添加其他类别
]

# 存储所有记录
scenes = []
samples = []
ego_poses = []
instances = []
sample_annotations = []

for scenario in scenarios:
    scene_token = f"nuplan_scene_{scenario.id}"
    scenes.append({
        "token": scene_token,
        "name": f"nuplan_{scenario.id}",
        "description": scenario.description,
        "log_token": "",  # 可留空
        "nbr_samples": len(scenario.frames),
        "first_sample_token": "",  # 稍后填充
        "last_sample_token": ""
    })

    # 为场景中的每个目标分配 instance
    # 需要跟踪所有出现的目标 ID，并创建 instance 条目
    # 假设 scenario 有所有 track 信息
    for track in scenario.tracks:
        instance_token = f"nuplan_instance_{track.id}"
        instances.append({
            "token": instance_token,
            "category_token": map_category(track.category),
            "nbr_annotations": len(track.frames),
            "first_annotation_token": "",
            "last_annotation_token": ""
        })

    # 遍历帧
    for frame_idx, frame in enumerate(scenario.frames):
        sample_token = f"nuplan_sample_{scenario.id}_{frame_idx}"
        ego_pose_token = f"nuplan_ego_{scenario.id}_{frame_idx}"
        timestamp = frame.timestamp  # 微秒

        # 自车位姿
        ego_pose = {
            "token": ego_pose_token,
            "timestamp": timestamp,
            "rotation": [frame.ego_pose.quaternion.w, frame.ego_pose.quaternion.x, frame.ego_pose.quaternion.y, frame.ego_pose.quaternion.z],
            "translation": [frame.ego_pose.translation.x, frame.ego_pose.translation.y, frame.ego_pose.translation.z]
        }
        ego_poses.append(ego_pose)

        # 样本
        samples.append({
            "token": sample_token,
            "timestamp": timestamp,
            "prev": "",  # 上一个 sample token，可后续填充
            "next": "",
            "scene_token": scene_token,
            "ego_pose_token": ego_pose_token
        })

        # 该帧中的目标标注
        for obj in frame.objects:
            annotation_token = f"nuplan_ann_{scenario.id}_{frame_idx}_{obj.id}"
            instance_token = f"nuplan_instance_{obj.track_id}"
            sample_annotations.append({
                "token": annotation_token,
                "sample_token": sample_token,
                "instance_token": instance_token,
                "category_token": map_category(obj.category),
                "attribute_tokens": [],  # 可选
                "visibility_token": "",  # 可选，可留空
                "translation": [obj.box.center.x, obj.box.center.y, obj.box.center.z],
                "size": [obj.box.width, obj.box.length, obj.box.height],
                "rotation": [obj.box.orientation.w, obj.box.orientation.x, obj.box.orientation.y, obj.box.orientation.z],
                "num_lidar_pts": 0,  # 可选
                "num_radar_pts": 0,
                "prev": "",
                "next": ""
            })

# 填充前后关系
# 需要根据 sample 的顺序设置 prev/next，以及 scene 的 first/last sample
# 同样填充 instance 的 first/last annotation
# 这部分逻辑需仔细实现
```

### 6.3 处理地图数据

MAPTR 使用向量化地图，通常以 `.pkl` 文件保存，包含车道线、人行横道等。nuPlan 的地图格式不同，需要转换为类似格式。

一种简单的方法是：将 nuPlan 的地图栅格化或提取矢量元素，保存为 nuScenes 地图的 `.pkl` 格式（如 `map_expansion.pkl`）。具体转换代码依赖地图 API，但可以提取车道中心线、边界等，保存为字典结构：

```python
map_dict = {
    'lane_centerline': [...],
    'lane_boundary': [...],
    'crosswalk': [...],
    ...
}
with open('nuplan_map.pkl', 'wb') as f:
    pickle.dump(map_dict, f)
```

如果不想转换地图，也可以在训练时忽略地图，但 MAPTR 的性能会受影响。

---

## 7. 合并数据集

转换完成后，你可以选择两种方式使用数据：

### 方式一：将 nuPlan 转换结果放入 nuScenes 目录

将生成的 JSON 文件与原始 nuScenes 的 JSON 文件合并（注意 token 不能冲突），地图文件也放入 nuScenes 的 `maps` 目录。这样 MAPTR 可以像加载 nuScenes 一样加载整个数据集。

### 方式二：保持分离，修改数据加载器

在 MAPTR 的 `nuscenes_dataset.py` 或数据加载部分，添加对 nuPlan 数据的支持，通过配置选择加载哪个数据集。这种方式更灵活，但需要修改代码。

推荐方式一，因为它对 MAPTR 代码改动最小。

---

## 8. 修改 MAPTR 配置

打开 MAPTR 的配置文件（如 `projects/configs/maptr/maptr_nuplan.py` 或类似），修改数据路径和类别映射：

```python
data_root = 'path/to/merged_nuscenes/'
ann_file = 'path/to/merged_nuscenes/v1.0-trainval/'
# 如果类别有变化，修改类别名称
class_names = ['vehicle', 'pedestrian', ...]
```

同时确保模型输入输出维度匹配（如预测轨迹点数、时间长度）。

---

## 9. 训练与验证

运行训练脚本：

```bash
python tools/train.py projects/configs/maptr/maptr_nuplan.py
```

验证时，如果希望单独评估 nuPlan 部分，可以根据场景 token 过滤样本。

---

## 10. 常见问题与注意事项

- **坐标系**：nuPlan 使用全局坐标系，nuScenes 也是全局坐标系，但原点可能不同。确保转换时保持一致性（一般直接使用全局坐标即可）。
- **时间同步**：nuPlan 帧率可能与 nuScenes 不同（nuPlan 通常 10Hz，nuScenes 2Hz）。在提取样本时，可能需要降采样或对齐时间。
- **类别映射**：nuPlan 的类别（如 `vehicle`, `pedestrian`, `bicycle`）与 nuScenes 基本一致，但需仔细对照。
- **轨迹提取**：MAPTR 通过滑动窗口从样本序列中提取未来轨迹。确保你的样本序列连续且时间间隔一致。
- **地图精度**：nuPlan 地图和 nuScenes 地图可能在不同坐标系或精度不同，如果影响训练，需进行坐标变换或重新提取。

---

## 总结

将 nuPlan 数据集集成到 MAPTR 中，本质上是一次数据格式转换和对齐的工作。通过将 nuPlan 数据转换为 nuScenes 格式，可以最大程度复用现有代码。虽然转换脚本需要根据数据细节调整，但整体流程清晰。希望本教程能帮助你顺利完成对齐。如有具体问题，欢迎进一步交流。