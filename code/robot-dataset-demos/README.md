# Robot Dataset Format Demos

独立子仓库：用最小样例理解机器人数据的 **HDF5 / LeRobot / RLDS** 三种格式，并对照 CV 里的 COCO/YOLO。

主仓库笔记：[数据格式术语对照与解析入门](https://github.com/EmbodiedDataHub/robot-manipulation-data-hub/blob/main/note/paper-note/Data-Pipeline/数据格式术语对照与解析入门.md)（路径以 submodule 挂载为准）

---

## 快速开始

```bash
cd code/robot-dataset-demos   # 子仓库根目录
pip install -r requirements.txt

# 1) 离线最小样例（已提交到仓库 / 或本地生成）
python scripts/create_minimal_samples.py
python scripts/inspect_hdf5.py samples/aloha_hdf5/episode_0.hdf5
python scripts/inspect_lerobot.py samples/lerobot_minimal

# 2) 下载真实公开小样
chmod +x scripts/download_samples.sh
./scripts/download_samples.sh --lerobot-only   # LeRobot ~67MB
./scripts/download_samples.sh --all            # + LIBERO + RLDS droid_100 (~2GB)

# RLDS 解析（需额外依赖）
pip install -r requirements-rlds.txt
python scripts/inspect_rlds.py   # 默认读 samples/tensorflow_datasets/droid_100/1.0.0
```

国内 HuggingFace 超时时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
./scripts/download_samples.sh --lerobot-only
```

---

## Notebook 教程（推荐）

交互式走通两种标准格式（相对本目录）：

```bash
cd code/robot-dataset-demos
pip install -r requirements.txt
jupyter notebook notebooks/01_lerobot_format.ipynb   # 先跑这个（无需 TF）
pip install -r requirements-rlds.txt
jupyter notebook notebooks/02_rlds_format.ipynb      # 需本机 droid_100 样例
```

| Notebook | 讲什么 | 默认样例 |
|----------|--------|----------|
| [`notebooks/01_lerobot_format.ipynb`](./notebooks/01_lerobot_format.ipynb) | 目录 / `info.json` / parquet 一帧 / 视频对齐 / 动作曲线 | `lerobot_aloha_sim_hf` 或 `lerobot_minimal` |
| [`notebooks/02_rlds_format.ipynb`](./notebooks/02_rlds_format.ipynb) | Episode→Steps / features 描述 / `action` vs `action_dict` / 图像 | `tensorflow_datasets/droid_100/1.0.0` |

---

## 目录结构

```text
robot-dataset-demos/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_lerobot_format.ipynb     # LeRobot 交互解析
│   └── 02_rlds_format.ipynb        # RLDS 交互解析
├── scripts/
│   ├── create_minimal_samples.py   # 离线生成 HDF5 + LeRobot 结构
│   ├── download_samples.sh         # 下载公开小样
│   ├── inspect_hdf5.py
│   ├── inspect_lerobot.py
│   └── inspect_rlds.py
└── samples/
    ├── aloha_hdf5/                 # 已提交：ALOHA 风格 HDF5
    ├── lerobot_minimal/            # 已提交：LeRobot parquet 结构
    ├── lerobot_aloha_sim_hf/       # 下载：真实 HF 数据（gitignore）
    ├── LIBERO/                     # 下载：HDF5 benchmark（gitignore）
    └── tensorflow_datasets/        # 下载：RLDS（gitignore）
```

---

## 脚本说明

| 脚本 | 作用 |
|------|------|
| `create_minimal_samples.py` | 无网络生成最小 HDF5 + LeRobot |
| `inspect_hdf5.py` | 打印 HDF5 树（ALOHA / robomimic） |
| `inspect_lerobot.py` | 读 `meta/info.json` + parquet |
| `inspect_rlds.py` | 读 TFDS RLDS（需 tensorflow） |
| `download_samples.sh` | 拉取公开小样 |

---

## 样例数据

| 路径 | 格式 | 来源 | 说明 |
|------|------|------|------|
| `samples/aloha_hdf5/` | HDF5 | 本仓库生成 | 已提交，~1MB |
| `samples/lerobot_minimal/` | LeRobot | 本仓库生成 | 已提交，parquet only |
| `samples/lerobot_aloha_sim_hf/` | LeRobot | [HF](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) | 需 `download_samples.sh` |
| `samples/LIBERO/` | HDF5 | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `--all` 时下载 |
| `samples/tensorflow_datasets/droid_100/1.0.0` | RLDS | GCS `gs://gresearch/robotics/droid_100` (~2GB) | 已下载可用 |

**RLDS 无本地数据**：用 [OXE Colab](https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb) 在线可视化。

---

## 对应官方代码库

| 格式 | 官方库 |
|------|--------|
| HDF5 (ALOHA) | [tonyzhaozh/aloha](https://github.com/tonyzhaozh/aloha) |
| HDF5 (robomimic) | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) |
| LeRobot | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| RLDS / OXE | [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment) |

---

## 作为子仓库使用

主仓库 `robot-manipulation-data-hub` 通过 git submodule 引用本仓库：

```bash
# 在主仓库根目录
git submodule update --init code/robot-dataset-demos
```

首次发布远程仓库（维护者）：

```bash
cd code/robot-dataset-demos
git remote add origin git@github.com:EmbodiedDataHub/robot-dataset-demos.git
git push -u origin main
```
