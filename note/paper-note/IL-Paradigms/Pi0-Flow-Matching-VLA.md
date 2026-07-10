---
title: "π0: A Vision-Language-Action Flow Model for General Robot Control"
authors: "Physical Intelligence"
year: 2024
source: "arXiv:2410.24164"
tags:
  - pi0
  - flow-matching
  - VLA
  - continuous-action
  - paradigm
pdf_path: "paper/Algorithm/Pi0/Pi0- A Vision-Language-Action Flow Model for General Robot Control.pdf"
related_notes:
  - "../Diffusion-Policy/概述.md"
  - "../RDT-Foundation-Models.md"
  - "./RT-2-Vision-Language-Action.md"
  - "./概述.md"
---

## 一句话总结

**π0** 代表 **连续动作 VLA 的 Gen-3 路线**：**PaliGemma VLM + 400M 级 Flow Matching 动作专家**，在 **10,000+ 小时多机器人 demo** 上训练，预测 **50 步 × 动作维** 的 action chunk，用 **velocity field MSE** 替代离散 token 或 DDPM——在 **精细、高频、多模态操作** 上优于离散 VLA，与 **RDT（Diffusion FM）** 并列的 **连续生成式 VLA** 范式。

---

## 1. 范式定位

| 轴 | π0 |
|----|-----|
| **A · 怎么学** | Offline BC（大规模多机器人） |
| **B · Action Head** | **Flow Matching**（连续 chunk） |
| **C · 数据** | 私有 **10K+ 小时** teleop（多 embodiment） |

```text
离散 VLA (RT-2/OpenVLA)  →  连续 Flow VLA (π0)  →  工业精细操作
         ↘                                    ↗
          Diffusion FM (RDT-1B) ─────────────
```

---

## 2. Flow Matching 是什么？

**与 Diffusion Policy 的关系**：同属 **生成式连续动作**；Flow Matching 学 **从噪声到数据的 velocity field**，训练更稳定、推理步数可更少。

| | DDPM (Diffusion Policy) | Flow Matching (π0) |
|---|-------------------------|---------------------|
| 训练目标 | 预测 noise ε | 预测 velocity $v_t$ |
| 采样 | 多步 denoise | ODE 积分（5–10 步） |
| 多模态 | ✅ | ✅ |
| 收敛 | 较慢 | 通常更快 |

**本地 PDF**：`paper/Algorithm/Flow-Matching/`（Chi et al. 教程，π0 理论基础）

---

## 3. 架构

```text
多视角 RGB + 语言指令
        ↓
   PaliGemma（VLM，~3B 级）
        ↓ 各层 hidden states
   Flow Matching Action Expert（~400M）
        ↓ 交叉注意力融合 VLM 特征
   输出: action chunk  A ∈ R^{50 × d_a}
```

| 模块 | 作用 |
|------|------|
| **PaliGemma** | 视觉-语言理解；预训练 web 知识 |
| **Action Expert** | 专门生成连续动作；与 RDT2 Stage 2 思路类似 |
| **Chunk** | H=50 步；50 Hz 级控制需求 |
| **条件** | VLM 多层特征 cross-attn |

**训练损失**：Flow Matching velocity MSE  
**推理**：~5–10 步 ODE 积分得到 action chunk

---

## 4. 数据

| 项 | 说明 |
|----|------|
| 规模 | **10,000+ 小时** 真机 teleop |
| 机器人 | 多种 arm + mobile manipulator |
| 任务 | 叠衣、清理、抓取等 **长时域精细** 任务 |
| 公开性 | 数据不公开；模型部分权重/细节社区复现中 |

**与 RDT2 对比**：π0 用 **机器人 teleop**；RDT2 强调 **纯 UMI 人类数据** 零样本跨 embodiment。

---

## 5. 实验要点

| 对比 | 结论 |
|------|------|
| vs OpenVLA / 离散 VLA | π0 在 **精细操作、长 horizon** 上更强 |
| vs Diffusion Policy scale | VLM 条件 + 更大数据 → 跨任务泛化 |
| 推理速度 | 比离散自回归快于多 token；UltraFast 蒸馏后更快（见 RDT2） |
| 局限 | 数据/算力门槛高；开源复现仍在进行 |

---

## 6. π0 vs RDT-1B vs RDT2

| | π0 | RDT-1B | RDT2 |
|---|-----|--------|------|
| 连续头 | **Flow Matching** | **Diffusion (DiT)** | RVQ + **Flow** |
| VLM | PaliGemma | T5-XXL（冻结） | Qwen2.5-VL |
| 预训练数据 | 私有 teleop | OXE 1M+ | UMI 10K hr |
| 双臂 | 支持 | **主打** | UMI 零样本 |
| Unified Action | 否 | **128-d** | EEF-centric |

---

## 7. 在 Action Head 演进中的位置

```text
Gen-2  离散 Token VLA     RT-2 / OpenVLA
Gen-3  连续生成式 VLA
         ├─ Flow Matching   π0          ← 本文
         ├─ Diffusion DiT   RDT-1B
         └─ 混合 RVQ+FM     RDT2
```

---

## 8. 推荐阅读

| 顺序 | 内容 |
|:--:|------|
| 1 | [Diffusion Policy](../Diffusion-Policy/概述.md) — 生成式动作基础 |
| 2 | Flow Matching 教程 §1–2 |
| 3 | **π0 全文** |
| 4 | [RDT](../RDT-Foundation-Models.md) — Diffusion FM 对照 |
| 5 | [RT-2](./RT-2-Vision-Language-Action.md) — 离散 VLA 对照 |

**自测**：
- π0 为什么不用离散 action token？（精细操作、量化误差）
- Flow Matching 和 DDPM 的训练目标有何不同？
- π0 的 action expert 如何接入 VLM？（cross-attn 多层特征）
