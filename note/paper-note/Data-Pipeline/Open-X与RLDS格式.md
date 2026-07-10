---
title: "Open X-Embodiment 与 RLDS 格式"
tags:
  - Open-X-Embodiment
  - RLDS
  - data-format
  - OXE
related_notes:
  - "../../数采到VLA训练-数据管线整体方案.md"
  - "../VLA-Datasets-Benchmarks-Data-Engines.md"
pdf_path: "paper/Surveys/Open-X-Embodiment-2310.08864.pdf"
---

## 一句话

**Open X-Embodiment (OXE)** 把 **22 种机器人、1M+ episodes** 统一到 **RLDS** 格式，是跨 embodiment **预训练数据的事实标准**；贡献新数据 = 实现 **TFDS DatasetBuilder** + 按 schema 导出 **observation / action / language**。

---

## 1. 为什么 OXE / RLDS 重要

| 问题 | OXE 回答 |
|------|---------|
| 各 lab 格式互不兼容 | **RLDS 统一 episode/step schema** |
| 单机器人数据不够 FM | **1M+ 混合预训练** |
| 跨 robot 迁移 | RT-1-X **+50%** 成功率（有条件） |

**论文**：arXiv:2310.08864 · 本地 `paper/Surveys/Open-X-Embodiment-2310.08864.pdf`

---

## 2. RLDS 逻辑结构

```text
tf.data.Dataset
 └── Episode
      ├── episode_metadata
      │    └── {file_path, robot_type, ...}
      └── steps (Dataset)
           ├── observation
           │    ├── image_{k}: uint8 [H,W,3]
           │    ├── state: float32 [D_state]
           │    └── natural_language_instruction: string (optional)
           ├── action: float32 [D_action]
           ├── discount, reward (often 0 for BC)
           ├── is_first, is_last, is_terminal
           └── language_instruction (step-level copy)
```

**特点**：

- **Ragged / padded steps** 均可
- **多相机** = 多个 `image_*` key
- **Action dim 因 dataset 而异**——训练时用 **normalization + embodiment embedding** 处理

---

## 3. 常见子数据集 Action 约定

| 子集 | Action 空间 | 维度 |
|------|------------|------|
| RT-1 / Bridge | Delta EEF + gripper | 7 |
| DROID | Absolute EEF + gripper | 7 |
| ALOHA 类 | Joint positions | 14 |
| 移动 manipulator | EEF + base | 8+ |

**OpenVLA 惯例**：导出时统一为 **7-d EEF delta** + quantile normalization。

**RDT 惯例**：映射到 **128-d unified**（见 [RDT 笔记 §4](../RDT-Foundation-Models.md)）

---

## 4. 贡献 OXE 的流程（工程）

```text
1. 采集 → Canonical IR（本团队内部）
2. 写 TFDS DatasetBuilder  subclass
3. 实现 _generate_examples():
     yield episode_id, {steps: generator}
4. action / state 填 float32 list
5. 图像 encode uint8 或 path reference
6. 注册 dataset name + version
7. 上传 GCS / 本地 TFRecord
8. 在 OXE mixing config 中注册 weight
```

**工具**：`tensorflow_datasets` · OXE 官方 GitHub · Colab 示例（见资源索引）

---

## 5. RLDS vs LeRobot

| | RLDS | LeRobot |
|---|------|---------|
| 生态 | TF / Jax 传统 | **PyTorch / HF** |
| 存储 | TFRecord | **Parquet + MP4** |
| 主要用途 | **OXE 预训练、跨 lab** | **微调、开源默认** |
| 本团队建议 | 贡献/预训练导出 | 日常采集与 ACT/DP 训练 |

**双轨策略**（[总方案 §4.1](../../数采到VLA训练-数据管线整体方案.md)）：

```text
Canonical IR ─┬─→ RLDS exporter（OXE 贡献）
              └─→ LeRobot exporter（HF 微调）
```

---

## 6. 混合训练注意事项

| 风险 | 对策 |
|------|------|
| **负迁移** | 采样权重 ∝ √N；dynamic reweight |
| Action 空间不一致 | Unified 128-d 或 per-dataset adapter |
| 控制频率不同 | 显式 **control_freq** 条件（RDT）或 resample |
| 语言质量差 | 重标注 pipeline |

---

## 7. 与其他数据集

| 数据集 | 与 OXE 关系 |
|--------|------------|
| RT-1 | OXE 重要子集 |
| BridgeData V2 | OXE 子集，低成本桌面 |
| DROID | OXE 子集，in-the-wild |
| DROID → OpenVLA | 常用 finetune 源 |

---

## 8. 阅读顺序

1. Open X-Embodiment 论文 **§3–4**  
2. [VLA 数据综述 §1.1](../VLA-Datasets-Benchmarks-Data-Engines.md)  
3. [OpenVLA 笔记](../IL-Paradigms/OpenVLA.md) — 如何消费 OXE  
4. [总方案 §4.2 / §9](../../数采到VLA训练-数据管线整体方案.md)

**自测**：为什么 OXE 允许不同 action dim 共存？训练时怎么处理？
