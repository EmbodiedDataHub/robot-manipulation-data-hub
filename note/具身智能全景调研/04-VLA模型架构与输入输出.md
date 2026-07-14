---
title: "主流 VLA 模型架构与输入输出"
tags:
  - VLA
  - architecture
  - OpenVLA
  - Octo
  - Pi0
  - RDT
  - RT-2
updated: "2026-07-14"
related_notes:
  - "../VLA算法层学习路线与论文清单.md"
  - "../paper-note/IL-Paradigms/概述.md"
  - "../paper-note/Data-Pipeline/预训练数据集清单与格式对齐.md"
---

# 主流 VLA 模型架构与输入输出

> **核心问题**：主流 VLA / Foundation Model 的网络结构是什么？训练时输入输出张量长什么样？action 如何表示？

---

## 一、总览：三代架构路线

```text
Gen-1  Robot Transformer     RT-1：EfficientNet + 离散 action，无 VLM
Gen-2  Token VLA              RT-2 / OpenVLA：VLM + action token（离散）
Gen-2.5 Modular Generalist    Octo：小 Transformer + 可换 action head
Gen-3  Continuous FM          π0 Flow / RDT DiT：连续 chunk + 生成式 head
Gen-3+ UMI-scale VLA          RDT2：7B VLM + RVQ/Flow，人类数据零样本迁机
```

**本地代码 submodule**（EmbodiedDataHub forks）：
- `code/openvla` · `code/octo` · `code/openpi` · `code/RoboticsDiffusionTransformer`

---

## 二、对比总表

| 模型 | 年份 | 参数量 | 视觉 | 语言 | Action Head | 预训练数据 | 格式 |
|------|------|--------|------|------|-------------|-----------|------|
| **RT-1** | 2022 | ~35M | EfficientNet-B3 | Embedding | 256-bin 离散 11D | 单机构 130K | RLDS |
| **RT-2** | 2023 | 55B/12B | PaLI ViT | PaLI LM | Action token（共享词表） | RT-1 + Web VLM co-train | RLDS |
| **OpenVLA** | 2024 | 7B | SigLIP So400M | Llama 2 7B | 7×256 离散 token | OXE ~970K | RLDS |
| **Octo** | 2024 | 27M–93M | 小 ViT | Task token | 离散 **或** Diffusion | OXE ~800K | RLDS |
| **π0** | 2024 | ~3B+400M | PaliGemma | PaliGemma | Flow Matching chunk H=50 | 私有 10K+h teleop | 内部 IR |
| **RDT-1B** | 2024 | 1.2B DiT | SigLIP | T5-XXL（冻结） | DDPM diffusion chunk | OXE 46 集 1M+ | RLDS→128D |
| **RDT2** | 2026 | 7B+400M | Qwen2.5-VL | Qwen2.5-VL | RVQ→Flow→蒸馏 | UMI 10K+h | UMI 格式 |

---

## 三、统一 I/O 抽象（读任何模型前先套这个壳）

```text
输入:
  I_t     多视角 RGB（有时 + 深度）
  ℓ       自然语言指令
  s_t     proprio（joint / EEF / unified state）
  meta    embodiment_id, control_hz, history...

输出:
  a_t     单步动作  或
  A_{t:t+H}  action chunk（ACT/DP/Flow/DiT 常见）

训练:
  BC loss = CE（离散）| MSE（回归）| ε-MSE（扩散）| v-MSE（Flow）
```

---

## 四、分模型详解

### 4.1 RT-1 · Robotics Transformer

**定位**：多任务 BC Transformer，**无 VLM**，VLA 前身的「纯机器人侧」。

```text
o_t = (RGB, instruction_embed, proprio)
        ↓
EfficientNet-B3 → TokenLearner → 8 image tokens
        ↓
Transformer（8 layers）
        ↓
11D discrete action（256 bins × 各维，含 base + arm + gripper）
```

| 维度 | 内容 |
|------|------|
| **输入** | 单 RGB；语言 FiLM；proprio |
| **输出** | 离散 token → 解码为 11D（移动臂 + 夹爪） |
| **数据** | Everyday Robots ~130K ep（OXE `fractal20220817_data`） |
| **笔记** | [RT-1](../paper-note/IL-Paradigms/RT-1-Robotics-Transformer.md) |

---

### 4.2 RT-2 · Vision-Language-Action

**定位**：**VLM + 机器人 co-train**；动作与文本共用 token 空间。

```text
(Web 图文 batch)  +  (Robot 图文 batch)
         ↓                    ↓
    PaLI-X / PaLI-3B     同 backbone
         ↓
  自回归预测下一 token
         ↓
  文本 token  或  action token（7D×bins）
```

| 维度 | 内容 |
|------|------|
| **输入** | 1–6 张图 + 文本 prompt |
| **输出** | 动作 token 串（与语言同词表） |
| **训练** | ~50% web / ~50% robot batch |
| **数据** | RT-1 数据 + 互联网 VLM 数据 |
| **笔记** | [RT-2](../paper-note/IL-Paradigms/RT-2-Vision-Language-Action.md) |

**与 OpenVLA**：同「离散 token VLA」族；RT-2 闭源 PaLI，OpenVLA 开源 SigLIP+Llama。

---

### 4.3 OpenVLA

**定位**：**开源 7B 离散 VLA**，OXE 预训练 + LoRA 微调生态。

```text
RGB (1–2 views) ──→ SigLIP ──→ vision tokens
                                    │
Instruction ──→ Llama 2 7B ────────┼── Causal LM
                                    ▼
                          7 action tokens / step
                          (7D EEF delta + gripper, 256 bins each)
```

| 维度 | 内容 |
|------|------|
| **输入** | 1–2 相机 RGB；文本指令；可选 proprio |
| **输出** | 7 个离散 token → 反量化 → **7D delta EEF** |
| **Action 对齐** | OXE 子集统一为 7D delta + **quantile 归一化** |
| **微调** | LoRA on Llama；vision 常冻结 |
| **代码** | `code/openvla` |
| **笔记** | [OpenVLA](../paper-note/IL-Paradigms/OpenVLA.md) |

**部署注意**：自回归 7 token/step；精细高频任务弱于连续 VLA。

---

### 4.4 Octo

**定位**：**轻量通用策略** + **模块化 finetune**；不必 7B VLM。

```text
Multi-view RGB + language/task token + proprio
              ↓
      Transformer backbone（共享）
              ↓
      readout token
         ┌────┴────┐
         ▼         ▼
    Discrete    Diffusion
     head         head
         ↓         ↓
    action      action chunk
    tokens      (连续)
```

| 维度 | 内容 |
|------|------|
| **输入** | 多视角图像；任务/语言 embedding；proprio |
| **输出** | 离散 token **或** diffusion 去噪轨迹（finetune 可换 head） |
| **规模** | Octo-Small ~27M · Octo-Base ~93M |
| **预训练** | OXE 精选 ~800K；**策展 delta EEF 子集为主** |
| **Finetune** | 数十–数百 demo；换 head 适配新 action dim |
| **代码** | `code/octo` |
| **笔记** | [Octo](../paper-note/IL-Paradigms/Octo.md) |

**设计哲学**：预训练不硬统一所有 action 语义，靠 **observation mask + finetune 换 head** 适配。

---

### 4.5 π0 · Flow Matching VLA

**定位**：**连续 action chunk + Flow Matching**；精细高频操作。

```text
多视角 RGB + 语言
      ↓
PaliGemma VLM (~3B)
      ↓ hidden states（多层 cross-attn）
Flow Matching Action Expert (~400M)
      ↓
A ∈ R^{50 × d_a}   # H=50 chunk, d_a 随 embodiment
```

| 维度 | 内容 |
|------|------|
| **输入** | 多相机 RGB；语言；proprio；多 embodiment 各自归一化 |
| **输出** | **50 步连续 action chunk** |
| **损失** | Flow velocity MSE |
| **推理** | 5–10 步 ODE 积分 |
| **数据** | 私有 10K+ 小时 teleop（未全公开） |
| **代码** | `code/openpi`（OpenPI 复现/微调） |
| **笔记** | [π0](../paper-note/IL-Paradigms/Pi0-Flow-Matching-VLA.md) |

**与 RDT**：同属连续 FM；π0 用 **Flow + PaliGemma**；RDT-1B 用 **DDPM DiT + T5/SigLIP**。

---

### 4.6 RDT-1B · Diffusion Foundation Model

**定位**：**双臂 + 跨机器人预训练** 的 DiT 扩散 FM。

```text
o_t = (3×RGB history, proprio z_t, language ℓ, control_hz c)
              ↓
SigLIP（图像）+ T5-XXL（语言，冻结）→ 条件向量
              ↓
DiT 1.2B（扩散 Transformer）
              ↓
去噪 → action chunk（128D unified space 的子集）
```

| 维度 | 内容 |
|------|------|
| **输入** | 3 路相机（2 帧历史）；proprio；T5 语言；**control frequency** |
| **输出** | 128D **unified action** 槽位（物理可解释填充 + pad 0 + **mask**） |
| **Unified Space** | 左臂 EEF 10 + 右臂 EEF 10 + 关节/gripper/base 等共 128 维 |
| **预训练** | OXE 46 数据集，1M+ 轨迹，√N 采样 + embodiment id |
| **微调** | ALOHA 6K+ 双臂轨迹 |
| **代码** | `code/RoboticsDiffusionTransformer` |
| **笔记** | [RDT](../paper-note/RDT-Foundation-Models.md) |

**128 槽示例**（概念）：`[L_EEF(10), R_EEF(10), L_joint(7), R_joint(7), ...]`，缺失维度 **mask=0** 而非仅 pad。

---

### 4.7 RDT2 · UMI-scale VLA

**定位**：**几乎纯 UMI 人类数据** + 7B VLM，**零样本跨 embodiment**。

```text
Stage 1  RVQ-VAE：连续动作 → 离散 code（对齐 VLM token）
Stage 2  Flow Matching：VLM hidden → 连续控制
Stage 3  单步蒸馏：降延迟部署
```

| 维度 | 内容 |
|------|------|
| **输入** | UMI 视角 RGB + 语言（Qwen2.5-VL） |
| **输出** | Relative EEF 7D chunk |
| **数据** | 10K+ 小时增强 UMI（100 设备 × 100+ 家庭） |
| **与 RDT-1B** | **不同路线**：RDT2 不做 128 槽 OXE 预训练主故事 |

---

## 五、Action 对齐四流派（与架构绑定）

读 I/O 时必须同时看 **action 语义**：

| 流派 | 代表 | 做法 |
|------|------|------|
| **A · 7D delta + quantile** | OpenVLA | 统一 EEF delta → 256 bins × 7 token |
| **B · 128D unified + mask** | RDT-1B | 物理槽位 + availability mask |
| **C · modular head** | Octo | 预训练 delta 子集；finetune 换 head |
| **D · continuous chunk + Flow** | π0 | 多 embodiment norm + H=50 chunk |

**逐步数值例子**：[预训练数据集清单 §三](../paper-note/Data-Pipeline/预训练数据集清单与格式对齐.md)

---

## 六、架构选型（工程视角）

| 场景 | 优先考虑 |
|------|----------|
| 单卡复现 7B VLA | OpenVLA + LoRA |
| 小模型快速 cross-robot finetune | Octo |
| 双臂精细 + OXE 预训练 | RDT-1B |
| 高频连续控制 / 叠衣类 | π0 / OpenPI |
| 无真机、UMI 规模化 | RDT2 路线 |
| Benchmark 基线（非 VLA） | ACT / Diffusion Policy |

---

## 七、LeRobot 生态 vs 上述 VLA

| 在 LeRobot 里 | 不在 LeRobot 原生 policy 里 |
|---------------|------------------------------|
| ACT, Diffusion, π0, SmolVLA, GR00T | OpenVLA, Octo, RDT（需各自 repo） |

LeRobot 是 **数据 + 训练框架**；OpenVLA/Octo/RDT 是 **独立模型仓库**，数据可经 [any4lerobot](https://github.com/Tavish9/any4lerobot) 转换。

---

## 八、学习资源

| 类型 | 链接 |
|------|------|
| 论文路线 | [VLA算法层学习路线](../VLA算法层学习路线与论文清单.md) |
| IL 范式索引 | [IL-Paradigms/概述](../paper-note/IL-Paradigms/概述.md) |
| 训练全貌 | [VLA训练与数据全貌-深度版](../VLA训练与数据全貌-深度版.md) |
| π0 博客 | [physicalintelligence.company/blog/pi0](https://www.physicalintelligence.company/blog/pi0) |
| OpenVLA 项目 | [openvla.github.io](https://openvla.github.io/) |
| Octo 项目 | [octo-models.github.io](https://octo-models.github.io/) |
| RDT 项目 | [rdt-robotics.github.io](https://rdt-robotics.github.io/) |

---

**自测**：
1. OpenVLA 一步推理输出几个 token？对应什么物理量？
2. RDT-1B 的 128D 与 OpenVLA 的 7D 本质差别是什么？
3. Octo 如何在预训练不硬统一 action 的情况下仍能混训？
