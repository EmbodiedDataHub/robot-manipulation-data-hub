---
title: "Octo: An Open-Source Generalist Robot Policy"
authors: "Octo Model Team (UC Berkeley / Stanford / Google)"
year: 2024
source: "arXiv:2405.12250"
tags:
  - Octo
  - VLA
  - generalist-policy
  - transformer
  - open-source
  - paradigm
pdf_path: "paper/Algorithm/Octo/Octo - An Open-Source Generalist Robot Policy.pdf"
related_notes:
  - "./OpenVLA.md"
  - "./RT-1-Robotics-Transformer.md"
  - "./概述.md"
---

## 一句话总结

**Octo** 提供 **开源通用机器人策略（Generalist Robot Policy）**：基于 **Transformer**，在 **OXE 等大规模多机器人数据** 上预训练，支持 **多种 action 表示（离散 token / diffusion head）** 和 **灵活的 finetune 协议**（新 robot、新相机、新动作用少量 demo 即可适配）——与 OpenVLA 并列的 **Gen-2.5 开源生态**，更强调 **policy 模块化和 cross-embodiment finetune**。

---

## 1. 范式定位

| 轴 | Octo |
|----|------|
| **A · 怎么学** | Offline BC，多任务多机器人混合 |
| **B · Action Head** | **可配置**：离散 token 或 **Diffusion head** |
| **C · 数据** | Open X-Embodiment + 其他开放数据集 |

```text
RT-1 (闭源多任务)  →  Octo (开源 generalist + 灵活 finetune)
```

---

## 2. 设计哲学

| 原则 | 含义 |
|------|------|
| **Generalist pretrain** | 一个 checkpoint 覆盖多种 robot / 任务 |
| **Modular finetune** | 新 embodiment 只需 **少量 demo + 标准协议** |
| **Action flexibility** | 同一框架可接离散或扩散头 |
| **轻量 base** | Octo-Small **~27M**，Octo-Base **~93M**（远小于 7B VLA） |

**与 OpenVLA 差异**：OpenVLA = **大 VLM + 离散 token**；Octo = **专用 Transformer policy + 模块化**，不依赖 7B LLM。

---

## 3. 架构（概念）

```text
Multi-view RGB + language/task token
        ↓
   Transformer backbone（共享）
        ↓
   ┌────┴────┐
   ▼         ▼
Discrete   Diffusion
 head        head（可选）
   ↓         ↓
 action     action chunk
 tokens     trajectory
```

| 输入 | 说明 |
|------|------|
| 图像 | 多视角，dataset-specific 分辨率 |
| 语言 | 任务描述 embedding |
| proprio | 关节 / EEF 状态 |

**训练**：标准 BC；离散头用 CE，扩散头用 denoise MSE。

---

## 4. Finetune 协议（论文核心贡献之一）

| 场景 | 做法 |
|------|------|
| 新 robot | 冻结大部分 backbone，训 action head + 少量 adapter |
| 新相机布局 | 数据增强 + 少量 finetune |
| 新 action dim | 替换 / 扩展 action head |
| 数据量 | **数十到数百 demo** 即可有可用策略 |

**意义**：降低 **cross-embodiment 迁移** 工程门槛，与 RDT2 零样本 UMI 是不同路线（Octo 仍需少量 finetune）。

---

## 5. 实验要点

| 对比 | 结论 |
|------|------|
| vs 从头 BC | Octo finetune **样本效率更高** |
| vs RT-1-X | 相近或更好（取决于任务） |
| vs OpenVLA | Octo 更小更快；OpenVLA 语言/语义更强 |
| 局限 | 无大 VLM → 开放词汇语义弱于 RT-2/OpenVLA |

---

## 6. 三代对照

| 代际 | 代表 | 特点 |
|------|------|------|
| Gen-1 | RT-1 | 闭源多任务 Transformer |
| Gen-2 | RT-2 | VLM + 离散 action |
| Gen-2.5 | **Octo / OpenVLA** | **开源** + OXE scale |
| Gen-3 | π0 / RDT | 连续 Flow/Diffusion FM |

---

## 7. 推荐阅读

1. [RT-1](./RT-1-Robotics-Transformer.md) — 多任务 Transformer 起源  
2. Octo §2–4 — finetune 协议  
3. [OpenVLA](./OpenVLA.md) — 大 VLA 路线对照  
4. [Open X 数据](../VLA-Datasets-Benchmarks-Data-Engines.md)  

**自测**：Octo 和 OpenVLA 的核心差异是什么？（专用 policy vs 7B VLM；模块化 vs 端到端 LLM）
