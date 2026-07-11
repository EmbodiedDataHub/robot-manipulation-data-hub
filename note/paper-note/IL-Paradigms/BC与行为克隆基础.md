---
title: "BC 与行为克隆 · 模仿学习基础范式"
tags:
  - behavioral-cloning
  - imitation-learning
  - paradigm
  - foundation
related_notes:
  - "../ALOHA/概述.md"
  - "../DAgger-Dataset-Aggregation.md"
  - "./概述.md"
---

## 一句话总结

**行为克隆（BC）** 是模仿学习最基础的范式：收集专家 demo $\{(o_t, a_t^*)\}$，用监督学习训练 $\pi_\theta(a|o)$，部署时策略独立执行。**CNNMLP** 是其典型实现（ResNet + MLP 逐步回归）；**ACT** 在其上加了 Chunk + CVAE。**所有 VLA/DP/RDT 的预训练目标本质上仍是 BC**——差别只在 action 怎么参数化、数据有多大规模。

---

## 1. 范式定义

```text
专家 demo:  D = {(o_t, a_t*)}  来自 teleop / 脚本 / 仿真
训练:       min_θ  E[ ℓ( π_θ(o_t), a_t* ) ]
部署:       a_t = π_θ(o_t)   每步闭环执行
```

| 符号 | 含义 |
|------|------|
| $o_t$ | 观测：图像、proprio、（可选）语言 |
| $a_t^*$ | 专家在该状态下的动作 |
| $\ell$ | 损失：MSE（连续）、CE（离散 token）、去噪 MSE（扩散） |

**BC 不是 RL**：没有 reward、没有与环境在线交互（除非 DAgger 等变体）。

---

## 2. 核心问题：误差累积（Compounding Error）

BC 训练分布 = **专家访问的状态** $d_{\pi^*}$  
BC 部署分布 = **策略自己访问的状态** $d_\pi$  

两者不等 → 小错累积 → 精细任务（插 USB、拧瓶盖）失败。

| 理论 | 结论 |
|------|------|
| Ross & Bagnell 2010 | 单步错误率 $\epsilon$ → 期望总错误 $\sim T^2\epsilon$ |
| 工程对策 | Action Chunk、闭环高频重规划、Diffusion 多模态、DAgger 扩数据 |

**本地精读**：[DAgger 笔记](../DAgger-Dataset-Aggregation.md)

---

## 3. BC 的两条改进路线（读 ALOHA 前必知）

### 路线 A · 时序（Action Chunking）

一次预测 $a_{t:t+H}$，每 $H$ 步才重新 query → 减少开环步数、鼓励轨迹一致。

- **代表**：ACT（$H=100$）、Diffusion Policy（horizon $H$）、π0（$H=50$）

### 路线 B · 多模态（Multimodal Actions）

同一 obs 下 demo 可能左绕或右绕；纯 MSE 学到「走中间」→ 撞墙。

| 方法 | 机制 |
|------|------|
| CVAE (ACT) | latent $z$ 采样不同模式 |
| Diffusion / Flow | 从噪声采样不同轨迹 |
| 离散 token | 自回归采样不同 token 序列 |

---

## 4. 代表实现对照

| 实现 | 网络 | 输出 | 损失 | 笔记 |
|------|------|------|------|------|
| **CNNMLP** | ResNet + MLP | $(B, 14)$ 单步 joint | MSE | [CNNMLP 导读](../ALOHA/CNNMLPPolicy-Code-Walkthrough.md) |
| **ACT** | ResNet + CVAE + Transformer | $(B, 100, 14)$ chunk | L1 + KL | [ACT 原理](../ALOHA/ACT-Model-Working-Principles.md) |
| **RT-1** | EfficientNet + Transformer | 离散 token | CE | [RT-1](./RT-1-Robotics-Transformer.md) |
| **Diffusion Policy** | UNet/Transformer + DDPM | $(H, d_a)$ 轨迹 | denoise MSE | [DP 概述](../Diffusion-Policy/概述.md) |

ALOHA 论文用 **CNNMLP 作消融**：证明 Chunk + CVAE 对精细双臂操作的必要性。

---

## 5. 选读：What Matters in Imitation Learning (Mosaic, 2021)

**arXiv**: 2106.00672 · 本地：`paper/IL-Action-Head/IL-Analysis/`

经验性结论（非新算法）：

| 因素 | 规律 |
|------|------|
| 数据量 | 更多 demo → 更高成功率，但边际递减 |
| 数据质量 | 一致、无 pause 的 demo 优于杂乱 demo |
| 数据增强 | 图像增强有帮助；动作噪声增强需谨慎 |
| 观测 | 多相机、proprio 通常必要 |

**何时读**：Phase 0 选读，建立「数据工程」直觉。

---

## 6. 在全局范式中的位置

```text
轴 A: BC 是 Offline IL 的默认起点
轴 B: CNNMLP → ACT → DP/Flow 是 action head 演进
轴 C: 单任务 teleop → 多任务 → FM 预训练
```

**BC 与 VLA 的关系**：VLA 训练 = BC +（通常）VLM 预训练权重 + 语言条件。RT-2/OpenVLA/π0 没有改「从 demo 学 action」的本质。

---

## 7. 推荐阅读顺序

1. [快速入门 Day 1–2](../../快速入门-最短学习路径.md) — IL 直觉  
2. [ALOHA 概述](../ALOHA/概述.md) — CNNMLP vs ACT  
3. [DAgger](../DAgger-Dataset-Aggregation.md) — 理论上限  
4. 回到 [IL 范式概览](./概述.md) Phase 1

---

## 8. 自测

- BC 的监督信号是什么？（demo 里的 action）
- ACT 解决了 BC 的哪两个问题？（compounding error 部分缓解 + 多模态）
- DAgger 改的是 learning paradigm 还是 action head？（paradigm）
