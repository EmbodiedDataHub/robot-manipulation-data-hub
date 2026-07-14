---
title: "具身智能全景调研 · 索引"
tags:
  - embodied-ai
  - survey
  - index
  - overview
updated: "2026-07-14"
---

# 具身智能全景调研

> **目标**：用 5 篇文档回答「数据 × 模型 × 格式 × 数据集 × 数采」五个核心问题，作为本仓库的**调研入口**。  
> **与现有笔记关系**：本系列是**汇总层**；深度细节链到 `paper-note/`、`VLA训练与数据全貌-深度版.md` 等。

---

## 文档清单

| # | 文档 | 回答什么问题 |
|---|------|-------------|
| 1 | [数据与模型双轴地图](./01-数据与模型双轴地图.md) | 以数据为中心 vs 以模型为中心，如何画具身智能整体地图 |
| 2 | [数据集格式标准与转换工具](./02-数据集格式标准与转换工具.md) | RLDS / LeRobot / HDF5 等标准、转换库、教程 |
| 3 | [公开数据集与下载方法](./03-公开数据集与下载方法.md) | OXE、DROID、Bridge、LIBERO 等及 gsutil/HF 命令 |
| 4 | [VLA 模型架构与输入输出](./04-VLA模型架构与输入输出.md) | RT / OpenVLA / Octo / π0 / RDT 的结构与 I/O |
| 5 | [数采方法分类与详解](./05-数采方法分类与详解.md) | Teleop / UMI / VR / Ego / 仿真 五条主线 |

---

## 推荐阅读顺序

```text
① 01 双轴地图（建立全局）
② 02 格式 + 03 数据集（数据工程师视角）
③ 05 数采（数据从哪来）
④ 04 VLA 架构（模型怎么消费数据）
```

**预计总时长**：通读 4–6 小时；配合本地 PDF 与代码 2–4 周。

---

## 本仓库关联入口

| 主题 | 更深文档 |
|------|----------|
| 快速入门 | [VLA与机器人整体认知地图](../VLA与机器人整体认知地图.md) |
| 训练全貌 | [VLA训练与数据全貌-深度版](../VLA训练与数据全貌-深度版.md) |
| 训练范式 | [VLA训练范式全景图](../VLA训练范式全景图.md) |
| 算法论文 | [VLA算法层学习路线与论文清单](../VLA算法层学习路线与论文清单.md) |
| 数据管线 | [数采到VLA训练-数据管线整体方案](../数采到VLA训练-数据管线整体方案.md) |
| 视频/博客 | [资源索引](../resources/links/资源索引.md) |
| 论文 PDF | [论文索引](../../paper/论文索引.md) |
| 代码 submodule | `code/openvla` · `code/octo` · `code/openpi` · `code/RoboticsDiffusionTransformer` · `code/robot-dataset-demos` |

---

## 外部持续更新资源

| 资源 | 链接 |
|------|------|
| VLA 数据集 & Benchmark 列表 | [github.com/ziyaow1010/vla-datasets-benchmarks](https://github.com/ziyaow1010/vla-datasets-benchmarks) |
| Open X-Embodiment | [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/) |
| LeRobot 官方文档 | [huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot) |
| Embodied-AI-Guide（中文百科） | [book/embodied-ai-guide](../../book/embodied-ai-guide/README.md) |
