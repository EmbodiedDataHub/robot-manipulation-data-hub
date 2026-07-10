---
title: "ACT / CNNMLP Policy 代码导读 · 索引"
source: "code/act/"
tags:
  - ACT
  - CNNMLP
  - index
related:
  - "CNNMLPPolicy-Code-Walkthrough.md"
  - "ACTPolicy-Code-Walkthrough.md"
code_path: "code/act"
---

# ACT / CNNMLP Policy 代码导读 · 索引

> **总入口**：[ALOHA 笔记集概述](./概述.md)

两篇已 **拆分**，请按顺序阅读：

| 顺序 | 文档 | 内容 |
|:----:|------|------|
| **1** | [**CNNMLPPolicy 代码导读**](./CNNMLPPolicy-Code-Walkthrough.md) | ResNet + MLP baseline · 6 层逐行拆解 · **先读** |
| **2** | [**ACTPolicy 代码导读**](./ACTPolicy-Code-Walkthrough.md) | chunk + CVAE + Transformer · 6 层逐行拆解 |

## 快速对照

| | CNNMLPPolicy | ACTPolicy |
|---|--------------|-----------|
| 输出 | 1 步 `(B,14)` | 100 步 `(B,100,14)` |
| 损失 | MSE | L1 + KL |
| 推理 | 每步 query | 每 100 步 query chunk |
| 模型 | `CNNMLP` | `DETRVAE` |

公共入口：`imitate_episodes.py` → `policy.py` → `detr/main.py` → `detr_vae.py`

---

## 以下为合并版存档（详细 ACT 节仍可用）

> 完整合并内容保留在下文，供检索；**新读者请优先打开上表两篇之一。**
