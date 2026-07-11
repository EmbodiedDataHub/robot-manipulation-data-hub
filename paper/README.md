# paper/ 目录说明

> 本地 PDF 论文库 · 与 [论文索引.md](./论文索引.md) 阅读顺序一致。

## 统一约定：每篇论文一个目录

**所有论文**（含综述、VLM、VLA、数据集）均遵循同一结构：

```
<分类>/<论文短名>/
├── <完整标题>.pdf
├── <完整标题>.zh.mono.pdf    # 中文 mono
└── <完整标题>.zh.dual.pdf    # 中英对照
```

示例：

```
VLA/VLM/SigLIP/
├── SigLIP- Sigmoid Loss for Language Image Pre-Training.pdf
├── SigLIP- Sigmoid Loss for Language Image Pre-Training.zh.mono.pdf
└── SigLIP- Sigmoid Loss for Language Image Pre-Training.zh.dual.pdf

VLA/RT-2/
├── RT-2- Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.pdf
├── …zh.mono.pdf
└── …zh.dual.pdf
```

## 顶层结构

```
paper/
├── Surveys/              领域综述（每篇一子目录）
├── VLA/                  Vision-Language-Action 全栈
│   ├── VLM/              Stage 0 · 视觉-语言底座（CLIP / SigLIP / LLaVA …）
│   ├── RT-1/             Stage 1+ · 策略与 Action Head
│   ├── RT-2/
│   ├── OpenVLA/
│   └── …
├── IL-Action-Head/       模仿学习 + 连续动作头（Diffusion / Flow / DAgger）
├── Datasets/             机器人数据集论文
├── Foundation-Models/    大规模 FM 预训练
└── Data-Collection/      数据采集方法与系统
    ├── ALOHA/
    ├── UMI/
    ├── VR-Humanoid/
    ├── Ego/
    ├── Video/
    └── Dexterous/
```

## 为什么 VLM 放在 `VLA/VLM/` 下？

VLA = **V**ision-**L**anguage-**A**ction。VLM 负责 **看懂 + 听懂**（VL），Action Head 负责 **动手**（A）。  
训练流水线是 **VLM 预训练 → 加载权重 → BC 训 Action**，因此 VLM 是 VLA 的 **Stage 0 子模块**，不是与 VLA 平行的独立领域。

```
VLA/
├── VLM/          ← 底座（通常冻结或 LoRA）
├── RT-2/         ← 离散 token VLA
├── OpenVLA/      ← 基于 Prismatic VLM
└── Pi0/          ← 基于 PaliGemma + Flow Head
```

## 分类速查

| 目录 | 放什么 |
|------|--------|
| **Surveys** | 综述、OXE、In-Hand 综述 |
| **VLA/VLM** | 图文对比、指令微调、Prismatic/PaliGemma |
| **VLA/**（除 VLM） | 输出动作为目标的 VLA / Robot Transformer |
| **IL-Action-Head** | DAgger、Diffusion Policy、Flow 教程 |
| **Datasets** | BridgeData、DROID、MimicGen |
| **Foundation-Models** | RDT、Qwen-RobotManip |
| **Data-Collection** | 数采硬件与协议 |

## 维护

```bash
# 更新索引
python scripts/generate_paper_index.py

# 批量翻译（输出写入同目录）
python scripts/batch_translate_paper.py
```

## 变更记录

- **2026-07-11**：`Surveys/`、`VLM/` 改为每篇一子目录；`VLM/` 迁入 `VLA/VLM/`
- **2026-07-11**：`Algorithm/` 拆为 `IL-Action-Head/` + `VLA/`；`Data Acquisition/` → `Data-Collection/`
