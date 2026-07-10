---
title: "RT-1: Robotics Transformer for Real-World Control at Scale"
authors: "Anthony Brohan, et al. (Google DeepMind)"
year: 2022
source: "arXiv:2212.06817"
tags:
  - RT-1
  - robotics-transformer
  - discrete-action
  - multi-task
  - imitation-learning
  - paradigm
pdf_path: "paper/Algorithm/RT-1/RT-1- Robotics Transformer for Real-World Control at Scale.pdf"
related_notes:
  - "./RT-2-Vision-Language-Action.md"
  - "./概述.md"
  - "../ALOHA/概述.md"
---

## 一句话总结

**RT-1** 用 **EfficientNet + TokenLearner + Transformer** 在 **700+ 任务、13 万条 demo** 上做多任务 BC，把连续 7-DoF 控制 **离散化成 256 bins × 7 维 → token 序列**，用 **交叉熵 next-token 预测** 训练——证明了 **一个 Transformer 可以 scale 到车队级真实机器人多任务控制**，是 **离散 Action Token 范式** 和后续 **RT-2 VLA** 的直接前身。

---

## 1. 研究动机

| 痛点 | RT-1 的回答 |
|------|------------|
| 每任务一个模型 | **一个模型 700+ 任务** |
| 小网络记不住 | **Transformer + 13 万 ep scale** |
| 连续动作难统一 | **离散化 → token，与 NLP 工具链兼容** |

**范式定位**：轴 A = Offline BC · 轴 B = **离散 Token** · 轴 C = **多任务单 embodiment（Everyday Robots）**

---

## 2. 任务形式化

```
输入 o_t:
  · 第三人称 RGB（320×256）
  · 自然语言指令 ℓ（任务级）
  · 本体 proprio（可选，7-d robot state）

输出 a_t:
  · 7 维: (x, y, z, roll, pitch, yaw, gripper)
  · 每维 uniform 离散化为 256 bins → 7 个 token / step
```

**控制频率**：约 3 Hz policy query（低频 chunk 思想的前身）。

---

## 3. 架构

```text
RGB ──→ EfficientNet-B3 ──→ TokenLearner ──→ 8 个 image tokens
                                                    │
Language ℓ ──→ USE embedding ──────────────────────┼──→ Transformer (8 layers)
                                                    │         │
                                                    ▼         ▼
                                              自回归预测 7 个 action tokens
```

| 模块 | 作用 |
|------|------|
| **EfficientNet-B3** | 视觉 backbone |
| **TokenLearner** | 把大量 spatial feature 压成 **8 个** compact token（算力关键） |
| **Transformer** | 多任务共享；causal self-attention |
| **Action tokenizer** | 连续 → 256 bins；decode 回连续控制 |

**训练损失**：$\mathcal{L} = \text{CrossEntropy}(\text{pred token}, \text{demo token})$

---

## 4. 数据

| 项 | 规模 |
|----|------|
| 任务数 | **700+** |
| Episodes | **~130K** |
| 机器人 | Everyday Robots 移动 manipulator |
| 采集 | 车队级 teleop |
| 成功率过滤 | 只用成功 demo |

**与 Open X 关系**：RT-1 数据是 OXE 合集的重要子集；RT-1-X 是在 OXE 上微调 RT-1 结构的实验。

---

## 5. 实验要点

- **多任务**：单模型在 held-out 任务上泛化优于 per-task BC
- **语言跟随**：FiLM / embedding 注入指令
- **泛化**：对新物体、新背景有一定零样本能力（仍有限）
- **局限**：无 VLM 预训练 → 语义/开放词汇能力弱于 RT-2

---

## 6. 与其它范式对照

| 维度 | RT-1 | ACT | Diffusion Policy | RT-2 |
|------|------|-----|------------------|------|
| Action | 离散 token | 连续 joint chunk | 连续轨迹 | 离散 token + VLM |
| 语言 | 任务 embedding | 无 | 无 | VLM 深度融合 |
| 多任务 | ✅ 700+ | 单任务为主 | 单任务为主 | ✅ + web 知识 |
| 预训练 | 无 VLM | 无 | 无 | PaLI co-train |

---

## 7. 关键概念（读 RT-2 前必记）

1. **Action discretization**：256 bins 是 OpenVLA 等开源 VLA 的默认设置来源  
2. **TokenLearner**：图像 → 少量 token，降低 Transformer 序列长度  
3. **多任务 BC**：所有任务混 batch 训练，instruction 区分任务  
4. **低频 control**：3 Hz vs ACT 50 Hz — 部署时需插值或 chunk

---

## 8. 推荐阅读

| 顺序 | 内容 |
|:--:|------|
| 1 | 本文 §1–4 + Fig 2 |
| 2 | [RT-2](./RT-2-Vision-Language-Action.md) — VLM 嫁接 |
| 3 | [Open X](../VLA-Datasets-Benchmarks-Data-Engines.md) — 数据格式 |
| 4 | [OpenVLA](./OpenVLA.md) — 开源复现 |

**自测**：7 维连续动作如何变成 7 个 token？为什么离散化有利于与 LLM 统一？
