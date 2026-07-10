---
title: "OpenVLA: An Open-Source Vision-Language-Action Model"
authors: "Moo Jin Kim, et al. (Stanford / UC Berkeley / TRI)"
year: 2024
source: "arXiv:2406.09246"
tags:
  - OpenVLA
  - VLA
  - discrete-action
  - open-source
  - LoRA
  - paradigm
pdf_path: "paper/Algorithm/OpenVLA/OpenVLA- An Open-Source Vision-Language-Action Model.pdf"
related_notes:
  - "./RT-2-Vision-Language-Action.md"
  - "./Octo.md"
  - "./概述.md"
---

## 一句话总结

**OpenVLA** 是 **RT-2 离散 Token VLA 路线的开源复现**：**Prismatic VLM（SigLIP + Llama 2 7B）** 在 **Open X-Embodiment ~970K episodes** 上 BC 训练，7 维动作 **×256 bins → 7 token/step**，支持 **LoRA 高效微调**——让研究者能在单卡/小集群上 **复现 7B VLA 训练与部署**，是 **Gen-2.5 开源 VLA 生态** 的代表。

---

## 1. 范式定位

| 轴 | OpenVLA |
|----|---------|
| **A · 怎么学** | Offline BC（多任务混合） |
| **B · Action Head** | **离散 Token**（与 RT-2 同族） |
| **C · 数据** | **Open X-Embodiment** 970K ep |

```text
RT-2 (闭源 PaLI)  →  OpenVLA (开源 SigLIP+Llama)  →  社区 LoRA finetune
```

---

## 2. 架构

```text
RGB (1–2 views) ──→ SigLIP ViT ──→ vision tokens
                                        │
Instruction ──→ Llama 2 7B ──────────┼──→ Causal LM
                                        │
                                        ▼
                              预测 7 个 action tokens / step
```

| 模块 | 说明 |
|------|------|
| **Prismatic VLM** | 视觉-语言对齐的开放 VLM 架构 |
| **SigLIP** | 视觉 encoder（So400M） |
| **Llama 2 7B** | 语言 backbone + action token 生成 |
| **Action bins** | 7 DoF × 256 bins，与 RT-2/Open X 惯例一致 |

**训练目标**：$\mathcal{L} = \text{CE}(\text{next token})$，action token 与 text token 同词表。

---

## 3. 数据与训练

| 项 | 配置 |
|----|------|
| 预训练数据 | OXE 子集 **~970K episodes** |
| 混合 | 多 robot、多任务、多 action space（经 dataset-specific 归一化） |
| 微调 | **LoRA** on Llama；vision 通常冻结 |
| 算力 | 多卡 A100 级（比 RT-2 55B 可及得多） |

**与 RDT 对比**：OpenVLA 走 **离散 + 开源 VLM**；RDT 走 **连续 Diffusion + DiT**，数据也偏 OXE 但 action 是 unified 128-d。

---

## 4. 实验要点

| 能力 | 表现 |
|------|------|
| OXE 内任务 | 接近或超过专用 BC baseline |
| 新 robot LoRA finetune | **7B 比 3B 小模型泛化更好**（论文 claim） |
| 精细操作 | 弱于连续 VLA（π0/RDT）——离散化瓶颈 |
| 推理速度 | 自回归 7 token，比 Flow 慢于单步 MLP |

---

## 5. 工程实践（读论文后建议做）

1. **OpenVLA GitHub** + **LeRobot** 加载 OXE 子集  
2. 理解 **action normalization** per dataset  
3. 在自己的 robot 上 **LoRA finetune** 50–200 demo  
4. 对比 **离散 OpenVLA vs 连续 ACT/DP** 在同一任务

---

## 6. 与其它 VLA 对照

| | OpenVLA | Octo | π0 | RDT-1B |
|---|---------|------|-----|--------|
| 开源 | ✅ | ✅ | 部分 | ✅ |
| 规模 | 7B | 93M–Octo-base | 3B+ | 1.2B DiT |
| Action | 离散 CE | 离散/扩散可选 | Flow | Diffusion |
| 数据 | OXE | OXE+ | 私有 10K hr | OXE 1M+ |
| 微调 | LoRA | 全量/adapter | 全量 | 130K 步 ft |

---

## 7. 推荐阅读

| 顺序 | 文档 |
|:--:|------|
| 1 | [RT-2](./RT-2-Vision-Language-Action.md) |
| 2 | OpenVLA §3–5 |
| 3 | [Octo](./Octo.md) — 另一开源路线 |
| 4 | [π0](./Pi0-Flow-Matching-VLA.md) — 连续 VLA 对照 |

**自测**：OpenVLA 和 RT-2 在 action 表示上有何相同？LoRA 微调通常冻结哪些模块？
