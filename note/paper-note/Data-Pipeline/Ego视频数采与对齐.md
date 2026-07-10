---
title: "Ego 视频数采与对齐"
tags:
  - ego
  - human-video
  - data-engine
  - retarget
  - data-collection
related_notes:
  - "../../数采到VLA训练-数据管线整体方案.md"
  - "../VLA-Datasets-Benchmarks-Data-Engines.md"
---

## 一句话

**Ego / 互联网人类视频** 是 **规模天花板最高、action 标签最弱** 的数据源：原始只有 RGB（+ 可选字幕），必须通过 **retarget / latent action / 仿真重建** 才能进入 VLA BC 管线——本仓库 **范式 D**，适合 **Tier 1 预训练 prior**，不能替代 robot-aligned 精调数据。

---

## 1. 原始数据形态

```text
输入:
  · 第一人称 MP4（Ego4D / 自采 / YouTube）
  · 可选: 字幕、音频、场景描述

缺失:
  · 机器人 action
  · 精确 6D 相机位姿（多数视频）
  · 与 deploy 机器人一致的观测空间
```

---

## 2. 本仓库 Ego 论文地图（13 篇）

| 优先级 | 论文 | 核心思路 | 本地 |
|:------:|------|---------|:----:|
| ⭐ | **EgoVLA** | 人类视频预训练 VLA | ✅ |
| ⭐ | **EgoScale** | 20k+ 小时 scaling law | ✅ |
| ⭐ | **Phantom** | 零 robot demo 路线探索 | ✅ |
| ⭐ | **EgoBridge** | OT 最优传输对齐 human↔robot | ✅ |
| ⭐ | **EgoMimic** | 人类 mimic → 机器人 | ✅ |
| ○ | EgoZero | 进一步 zero-shot 探索 | ✅ |
| ○ | EgoHumanoid / EgoMI / EMMA | 人形/移动/主动视觉 | ✅ |
| ○ | Human→Robot Transfer | 何时涌现迁移 | ✅ |
| ○ | MotionTrans | VR 人体 → robot motion | ✅ |

目录：`paper/Data Acquisition/Ego Human Data/`

---

## 3. 四条对齐路径（→ 可用训练数据）

| 路径 | 代表 | 管线 | 输出 fidelity |
|------|------|------|:-------------:|
| **结构重建** | Video2Policy | VLM 语义 → 仿真任务代码 → rollout | ⭐⭐⭐ |
| **视觉 inpainting** | H2R | 人手→机械手，提取视觉运动 | ⭐⭐ |
| **物理 retarget** | RoboWheel, **EgoBridge** | 手 pose → SDF/OT 优化 | ⭐⭐⭐ |
| **Latent action** | Phantom, EgoVLA | 不直接 motor command | ⭐（预训练用） |

详见 [VLA 数据综述 §3.1 / §深入四种数采](../VLA-Datasets-Benchmarks-Data-Engines.md)

---

## 4. 进入 Canonical IR 的方式

### 4.1 强监督（retarget 成功）

```yaml
source: ego_retarget
observations:
  images: {ego_rgb: ...}      # 原始或 inpaint 后
action:
  eef_delta: [...]            # retarget 输出，需 QA
supervision: strong
```

### 4.2 弱监督（仅预训练）

```yaml
source: ego_latent
observations:
  images: {ego_rgb: ...}
action:
  latent_id: int              # 或 placeholder
supervision: weak             # 只用于 representation pretrain
```

**规则**：weak supervision 样本 **不得** 与 Tier 3 teleop 混 batch 做 final BC，除非有明确 multi-stage recipe。

---

## 5. Data Pyramid 中的位置

```text
Tier 1 · Ego 20k hr     → 语义/affordance prior（EgoScale）
Tier 2 · UMI + OXE      → 可执行 action 预训练
Tier 3 · Teleop 数百 hr → 目标 robot 精调（必须）
```

**2026 共识**（[综合调研报告 §3.D](../../机器人数据工作综合调研报告.md)）：

> Ego 是 **预训练 prior**，Finetune 必须补 **robot-aligned** 数据。

---

## 6. 前处理 QA（Ego 特有）

| 检查 | 规则 |
|------|------|
| 手部可见性 | 遮挡 > 50% 帧 → 降权或丢弃 |
| Retarget 误差 | 仿真/SDF 碰撞惩罚超阈值 → 丢弃 |
| 时序一致性 | 光流/跟踪断裂 → 分段 |
| 语义标注 | VLM 生成指令 + 10% 人工 spot check |

---

## 7. 阅读顺序

1. [VLA 数据综述](../VLA-Datasets-Benchmarks-Data-Engines.md) — Video-to-Data 引擎  
2. **EgoScale** — scaling 规律  
3. **EgoBridge** — OT 对齐方法  
4. **EgoVLA** — 如何接入 VLA 训练  
5. [总方案 §3.5 / §6.6](../../数采到VLA训练-数据管线整体方案.md)

---

## 8. 与 UMI 对比

| | UMI | Ego 视频 |
|---|-----|----------|
| 硬件 | 特制夹爪+GoPro | 无 / 普通相机 |
| Action | SLAM 推断 EEF | 需 retarget/重建 |
| Fidelity | ⭐⭐⭐⭐ | ⭐⭐ |
| Scale | 中→大 | **极大** |
| 部署 | 同构夹爪迁移 | 需额外 robot 校准 |
