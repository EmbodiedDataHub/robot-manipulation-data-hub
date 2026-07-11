---
title: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
authors: "Anthony Brohan, et al. (Google DeepMind)"
year: 2023
source: "arXiv:2307.15818"
tags:
  - RT-2
  - VLA
  - vision-language-action
  - discrete-action
  - co-training
  - paradigm
pdf_path: "paper/VLA/RT-2/RT-2- Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.pdf"
related_notes:
  - "./RT-1-Robotics-Transformer.md"
  - "./OpenVLA.md"
  - "./概述.md"
---

## 一句话总结

**RT-2** 正式定义 **VLA（Vision-Language-Action）**：把 **PaLI-X / PaLM-E 视觉-语言模型** 与 RT-1 机器人数据 **co-train**，将机器人动作 **表示为与文本相同的离散 token**，用 **next-token 交叉熵** 联合训练——使机器人能利用 **互联网 VLM 知识**（如认识「已灭绝动物」玩具），是 **离散 Token VLA 范式** 的开山之作。

---

## 1. VLA 是什么？

> **输入**：图像 + 自然语言指令  
> **输出**：动作（与文本一样，是一串 token）  
> **训练**：机器人 demo BC +（可选）web 视觉-语言数据 co-train

```text
「pick up the extinct animal」+ 图片
        ↓
   PaLI-X Transformer
        ↓
  token 序列: [..., action_tok_142, action_tok_87, ...]
        ↓
  decode → 7-DoF 机器人命令
```

**范式定位**：轴 A = BC + **web co-train** · 轴 B = **离散 Token（VLM 词表）** · 轴 C = RT-1 数据 + **互联网 VLM 预训练**

---

## 2. 核心创新

### 2.1 Action Tokenization（与 RT-1 相同思路，嵌入 VLM）

| 步骤 | 操作 |
|------|------|
| 1 | 连续动作 7 维，每维 256 bins |
| 2 | 映射到 **VLM 词表中 reserved token** |
| 3 | 与文本 token 同一序列，causal LM 训练 |

**好处**：架构统一；VLM 的语义知识可迁移到控制  
**代价**：量化误差；长 horizon 自回归慢

### 2.2 Co-training

| 数据类型 | 作用 |
|----------|------|
| RT-1 机器人 demo | 学「怎么做」 |
| Web 视觉-语言（PaLI 原有） | 保持「看懂、听懂」 |

**关键**：联合训练防止 VLM 能力在机器人微调中 **灾难性遗忘**。

### 2.3 涌现能力（Emergent Capabilities）

训练后 RT-2 展现出 **纯 RT-1 没有** 的能力：

- 识别训练集未见的语义类别（「extinct animal」→ 拿对应玩具）
- 简单推理（「把物体放到数字 2 上」）
- 跨语言指令（部分）

**直觉**：VLM 的 web 知识 **通过共享 token 空间** 泄漏到 action 生成。

---

## 3. 模型变体

| 模型 | 底座 | 规模 |
|------|------|------|
| RT-2-PaLI-X | PaLI-X | 55B（论文主要报告） |
| RT-2-PaLM-E | PaLM-E | 12B / 562B 等 |

**输入模态**：1–6 张图 + 文本；与 PaLI 格式一致。

---

## 4. 训练与推理

| 项 | 配置 |
|----|------|
| 损失 | Causal LM CrossEntropy（文本 + action token） |
| 微调 | 在 RT-1 数据上继续 co-train |
| 推理 | 自回归生成 action token → bin decode |
| 频率 | 低频控制（~1–3 Hz 级） |

---

## 5. 实验要点

| 设置 | 结果 |
|------|------|
| Seen tasks | 接近或超过 RT-1 |
| Unseen objects / backgrounds | **显著优于 RT-1** |
| 语义指令 | 开放词汇 pick 提升明显 |
| 局限 | 精细操作、高频控制仍弱；依赖离散化精度 |

---

## 6. 三代 VLA 谱系中的位置

```text
RT-1 (纯机器人 Transformer, 离散 action)
  ↓ + VLM co-train
RT-2 (VLA 定义, 离散 action token)     ← 本文
  ↓ 开源复现
OpenVLA / Octo (7B 级开源 VLA)
  ↓ 连续动作回归
π0 / RDT (Flow / Diffusion FM)
```

---

## 7. 与其它范式对照

| | RT-2 | OpenVLA | π0 | ACT |
|---|------|---------|-----|-----|
| 底座 | PaLI-X | SigLIP+Llama | PaliGemma | ResNet+Transformer |
| Action | 离散 token | 离散 token | Flow 连续 | 连续 CVAE |
| Co-train web | ✅ | 部分 | ✅ | ❌ |
| 开源 | ❌ | ✅ | 部分 | ✅ |

---

## 8. 推荐阅读顺序

1. [RT-1](./RT-1-Robotics-Transformer.md) — 离散 action 起源  
2. **RT-2 全文** — VLA 定义  
3. [Open X](../VLA-Datasets-Benchmarks-Data-Engines.md) — 数据基础设施  
4. [OpenVLA](./OpenVLA.md) — 开源离散 VLA  
5. [π0](./Pi0-Flow-Matching-VLA.md) — 连续 VLA 对照  

**自测**：
- RT-2 的 action 和 RT-1 的 action 表示有何相同？（256 bins → token）
- RT-2 比 RT-1 多的是什么？（VLM 预训练 + co-train + 语义涌现）
- 离散 VLA 的两个主要缺点？（量化误差、推理慢）
