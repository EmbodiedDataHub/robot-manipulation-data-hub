# VLA 算法层 · 学习路线与论文清单

> **目标**：按 **算法层** 建立阅读顺序，从模仿学习基础 → Action Head → VLA 主线 → Foundation Model。  
> **状态**：标注本地是否已收录；**论文下载与翻译后续批量进行**。  
> **关联**：[**VLA 训练范式全景图（IL+RL）**](./VLA训练范式全景图.md) · [**IL 范式概览**](./paper-note/IL-Paradigms/概述.md) · [训练与数据全貌（深度版）](./VLA训练与数据全貌-深度版.md) · [论文索引](../paper/论文索引.md) · [多媒体资料索引](./resources/links/资源索引.md)（视频 / 博客 / 代码）

---

## 使用说明

| 符号 | 含义 |
|------|------|
| ⭐ | 主线必读 |
| ○ | 选读 / 进阶 |
| ✅ | 本地 `paper/` 已收录 |
| ⬜ | **待下载**（见 [§八 待下载清单](#八待下载论文批量任务)） |
| 🔤 | 已有中文翻译 |

**推荐总时长**：算法层 **4–6 周**（每天 1–2 小时），与数据层可并行。

---

## 学习路线总览（7 层）

```
Layer 0  机器人学习基础     BC · 误差累积 · Sim2Real
    ↓
Layer 1  Action 表示与 Chunk  ACT · Diffusion Policy
    ↓
Layer 2  多任务 Robot Transformer  RT-1 · Perceiver-Actor
    ↓
Layer 3  VLA 开山            RT-2 · PaLM-E · Open X-Embodiment
    ↓
Layer 4  开源 VLA 生态        OpenVLA · Octo · RoboCat
    ↓
Layer 5  连续动作 VLA         π0 · RDT · Flow Matching
    ↓
Layer 6  综述与全貌           VLA Anatomy · Systematic Review
    ↓
Layer 7  前沿                 世界模型 · Ego→VLA · RL post-training
```

---

## Layer 0 · 机器人学习基础（3–5 天）

**本层要搞懂**：监督学习式模仿、误差累积、为什么需要 Chunk / Diffusion。

| 序 | 优先级 | 论文 | arXiv | 本地 | 译 | 读什么 | 搞懂什么 |
|:--:|:------:|------|-------|:----:|:--:|--------|---------|
| 0.1 | ⭐ | **ALOHA** (含 ACT) | [2304.13705](https://arxiv.org/abs/2304.13705) | ✅ | 🔤 | Method + ACT 段 | Action Chunking、CVAE、双臂 joint space · **视频** [B站精析](https://www.bilibili.com/video/BV1xGF3eeEjB) |
| 0.2 | ⭐ | **Diffusion Policy** | [2303.04137](https://arxiv.org/abs/2303.04137) | ⬜ | — | §1–3 + Fig | 多模态动作、denoise 出轨迹 |
| 0.3 | ○ | What Matters in IL (Mosaic) | [2106.00672](https://arxiv.org/abs/2106.00672) | ⬜ | — | Abstract | 数据量/质量/增强的经验结论 |
| 0.4 | ○ | DAgger | [1011.0686](https://arxiv.org/abs/1011.0686) | ✅ | — | §1–2 | 误差累积的 classic 解法（了解即可） → [[DAgger-Dataset-Aggregation\|精读笔记]] |

**本地笔记**：[ALOHA 笔记集概述](./paper-note/ALOHA/概述.md) · [CNNMLPPolicy 代码导读](./paper-note/ALOHA/CNNMLPPolicy-Code-Walkthrough.md) · [ACT 原理](./paper-note/ALOHA/ACT-Model-Working-Principles.md) · [ACTPolicy 代码导读](./paper-note/ALOHA/ACTPolicy-Code-Walkthrough.md) · **视频**：[B站 ACT/ALOHA 精析](https://www.bilibili.com/video/BV1xGF3eeEjB)（~42 min）

**自测**：
- BC 和 VLA 训练目标有何相同？（都是 (obs, lang) → action 的监督学习）
- 为什么 Diffusion 比 MLP 更适合「左绕/右绕都行」的任务？

---

## Layer 1 · Action 表示与 Policy 架构（1 周）

**本层要搞懂**：动作怎么参数化、Transformer 怎么做 policy、Diffusion 怎么接 vision。

| 序 | 优先级 | 论文 | arXiv | 本地 | 译 | 读什么 | 搞懂什么 |
|:--:|:------:|------|-------|:----:|:--:|--------|---------|
| 1.1 | ⭐ | **Diffusion Policy** | 2303.04137 | ⬜ | — | Method 全文 | Visual conditioning、horizon H、推理 loop |
| 1.2 | ⭐ | **ACT** (见 ALOHA) | 2304.13705 | ✅ | 🔤 | ACT 节 | Chunk size、k=100、CVAE latent |
| 1.3 | ○ | 3D Diffusion Policy | [2406.01586](https://arxiv.org/abs/2406.01586) | ⬜ | — | Abstract | 点云/深度作为 obs 的 diffusion |
| 1.4 | ○ | Chi et al. Flow Matching 综述 | [2404.08427](https://arxiv.org/abs/2404.08427) | ⬜ | — | §1–2 | π0 的理论基础 |

**与数据层的连接**：读 UMI 时对照——为什么 relative delta + Diffusion Policy 是黄金组合。

---

## Layer 2 · 多任务 Robot Transformer（1 周）

**本层要搞懂**：RT-1 如何把 700+ 任务合成一个 Transformer；Action tokenization 的前身。

| 序 | 优先级 | 论文 | arXiv | 本地 | 译 | 读什么 | 搞懂什么 |
|:--:|:------:|------|-------|:----:|:--:|--------|---------|
| 2.1 | ⭐ | **RT-1: Robotics Transformer** | [2212.06817](https://arxiv.org/abs/2212.06817) | ⬜ | — | §1–4 + Fig 2 | TokenLearner、离散 action、多任务 |
| 2.2 | ○ | **BC-Z** (Google) | [2203.02827](https://arxiv.org/abs/2203.02827) | ⬜ | — | Abstract | 语言条件 + BC 的早期大规模尝试 |
| 2.3 | ○ | **Gato** | [2205.05176](https://arxiv.org/abs/2205.05176) | ⬜ | — | §2 | 通用 token 序列（机器人只是模态之一） |

**关键概念**：
- **EfficientNet-B3 + TokenLearner** → 图像压成少量 token
- **Action discretization**：连续控制 → 256 bins
- **Instruction embedding**：FiLM 或 cross-attention 注入语言

---

## Layer 3 · VLA 开山（1–2 周）

**本层要搞懂**：VLM + 机器人数据；co-training；Open X-Embodiment 为何是基础设施。

| 序 | 优先级 | 论文 | arXiv | 本地 | 译 | 读什么 | 搞懂什么 |
|:--:|:------:|------|-------|:----:|:--:|--------|---------|
| 3.1 | ⭐ | **RT-2: VLA** | [2307.15818](https://arxiv.org/abs/2307.15818) | ⬜ | — | 全文 | **VLA 定义**、action token、web co-train |
| 3.2 | ⭐ | **Open X-Embodiment** | [2310.08864](https://arxiv.org/abs/2310.08864) | ✅ | 🔤 | §3–4 | RLDS、跨 robot 迁移、RT-1-X |
| 3.3 | ⭐ | **PaLM-E** | [2303.03378](https://arxiv.org/abs/2303.03378) | ⬜ | — | §3 | 多模态 embodied 输入嵌入 LLM |
| 3.4 | ○ | **VIMA** | [2210.03094](https://arxiv.org/abs/2210.03094) | ⬜ | — | Method | 结构化 language + token 化操作 |

**读完应能回答**：
- RT-2 的 action 怎么变成 token？
- 为什么需要 OXE 而不是只用 RT-1 数据？

---

## Layer 4 · 开源 VLA 生态（1–2 周）

**本层要搞懂**：2024 开源复现路线；LoRA finetune；OXE 上训 7B VLA。

| 序 | 优先级 | 论文 | arXiv | 本地 | 译 | 读什么 | 搞懂什么 |
|:--:|:------:|------|-------|:----:|:--:|--------|---------|
| 4.1 | ⭐ | **OpenVLA** | [2406.09246](https://arxiv.org/abs/2406.09246) | ⬜ | — | §3–5 | Prismatic VLM、离散 action、LoRA |
| 4.2 | ⭐ | **Octo** | [2405.12250](https://arxiv.org/abs/2405.12250) | ⬜ | — | §2–4 | Transformer policy、finetune 协议 |
| 4.3 | ○ | **RoboCat** | [2306.11706](https://arxiv.org/abs/2306.11706) | ⬜ | — | Abstract | 自改进 data flywheel |
| 4.4 | ○ | **TinyVLA / MiniVLA** | 搜 2024–2025 | ⬜ | — | — | 小模型部署 |

**实践**：OpenVLA GitHub + LeRobot 加载 OXE 子集 → 理解 dataloader。

---

## Layer 5 · 连续动作 VLA / Flow（1–2 周）

**本层要搞懂**：为什么工业界转向 Flow Matching；Unified Action Space。

| 序 | 优先级 | 论文 | arXiv | 本地 | 译 | 读什么 | 搞懂什么 |
|:--:|:------:|------|-------|:----:|:--:|--------|---------|
| 5.1 | ⭐ | **π0 (Pi-Zero)** | [2410.24164](https://arxiv.org/abs/2410.24164) | ⬜ | — | 全文 | Flow matching、PaliGemma、50Hz |
| 5.2 | ⭐ | **RDT-1B** | 见本地 | — | ✅ | 🔤 | Method | DiT、Unified Action、双臂 · **视频** [B站解读](https://www.bilibili.com/video/BV1FjyHYmEDQ) · [笔记](./paper-note/RDT-Foundation-Models.md) |
| 5.3 | ⭐ | **RDT2** | 见本地 | — | ✅ | 🔤 | 数据+模型 | 规模化 recipe · [笔记](./paper-note/RDT-Foundation-Models.md#part-b-rdt2) |
| 5.4 | ⭐ | **Qwen-RobotManip** | 见本地 | — | ✅ | 🔤 | 数据管线 | 中文 VLA 数据工程 |
| 5.5 | ○ | **GR00T N1** | NVIDIA 2025 | ⬜ | — | Technical report | 人形 whole-body VLA |

**对照阅读**：[训练全貌 §7 Recipe 对照](./VLA训练与数据全貌-深度版.md#七三种典型训练-recipe对照表)

---

## Layer 6 · 综述（建立算法+数据统一视图）（1 周）

**本层要搞懂**：领域全貌、开放问题、evaluation gap。

| 序 | 优先级 | 论文 | arXiv | 本地 | 译 | 读什么 |
|:--:|:------:|------|-------|:----:|:--:|--------|
| 6.1 | ⭐ | **VLA Datasets, Benchmarks, Data Engines** | 2604.23001 | ✅ | 🔤 | 全文（数据视角） |
| 6.2 | ⭐ | **VLA Anatomy Survey** | 2512.11362 | ✅ | 🔤 | §2–4（模型视角） |
| 6.3 | ⭐ | **VLA Systematic Review** | 2507.10672 | ✅ | 🔤 | Training pipeline 章 |
| 6.4 | ○ | **Large VLA Models Survey** | 2508.13133 | ✅ | 🔤 | 模型对比表 |
| 6.5 | ○ | **Towards Unified Robot Manipulation** | 2510.10903 | ✅ | 🔤 | 操作全领域索引 |

**本地笔记**：[VLA 数据综述笔记](./paper-note/VLA-Datasets-Benchmarks-Data-Engines.md)

---

## Layer 7 · 前沿（选读）

| 序 | 主题 | 论文 | arXiv | 本地 | 说明 |
|:--:|------|------|-------|:----:|------|
| 7.1 | Ego → VLA | **EgoVLA** | 见本地 | ✅ | 人类视频预训练 |
| 7.2 | Ego scaling | **EgoScale** | 见本地 | ✅ | 20k 小时 scaling law |
| 7.3 | H2R transfer | **Human→Robot Transfer in VLA** | 见本地 | ✅ | 何时涌现迁移 |
| 7.4 | 世界模型 | **UniSim / 3D-VLA** | 搜 2024 | ⬜ | 视频预测式训练 |
| 7.5 | RL post-train | **RLAIF-V / RoboRL** | 搜 2025 | ⬜ | 部署后优化 |

---

## 按周计划（算法层专用）

| 周 | Layer | 核心产出 |
|:--:|-------|---------|
| 1 | L0 + L1 | 能解释 BC / Chunk / Diffusion 区别；跑通 Diffusion Policy Push-T 或读透博客 |
| 2 | L2 + L3 | 画出 RT-1 → RT-2 架构演进；理解 action tokenization |
| 3 | L4 | 理解 OpenVLA 训练配置；知道 LoRA finetune 流程 |
| 4 | L5 | 对比 π0 vs RDT vs OpenVLA 三种 Recipe |
| 5 | L6 | 综述串联；写出「数据×算法」双轴地图 |
| 6 | L7 + 实践 | LeRobot finetune 一个 checkpoint |

---

## 算法 × 数据 对照矩阵

读算法论文时，**同时问数据问题**：

| 论文 | 算法要点 | 数据从哪来 | Action 空间 | 预训练规模 |
|------|---------|-----------|------------|-----------|
| ACT/ALOHA | CVAE + Chunk | 自建 teleop | 14-d joint | 每任务 50–200 ep |
| Diffusion Policy | DDPM action | sim + real teleop | EEF delta | 50–300 demo |
| RT-1 | Transformer | Everyday Robots 车队 | discretized EEF | 130K ep |
| RT-2 | VLM + co-train | RT-1 + web VLM | action token | 同 RT-1 + web |
| OpenVLA | SigLIP+Llama | OXE 970K | 7×256 bins | 970K ep |
| π0 | Flow matching | 私有多机器人 | continuous chunk | 10K+ hrs |
| RDT | DiT diffusion | OXE + 自采 | unified 128-d | 1M+ ep |

---

## 八、算法层论文库（已下载）

以下 **15 篇** 已收录于 [`paper/VLA/`](../paper/论文索引.md#0-algorithm)，**中文 mono + dual 已全部生成**（2026-07-07）。

| # | 论文 | arXiv | 目录 | 本地 | 译 |
|---|------|-------|------|:----:|:--:|
| 1 | Diffusion Policy | 2303.04137 | `IL-Action-Head/Diffusion-Policy/` | ✅ | 🔤 |
| 2 | RT-1 | 2212.06817 | `VLA/RT-1/` | ✅ | 🔤 |
| 3 | RT-2 | 2307.15818 | `VLA/RT-2/` | ✅ | 🔤 |
| 4 | PaLM-E | 2303.03378 | `VLA/PaLM-E/` | ✅ | 🔤 |
| 5 | OpenVLA | 2406.09246 | `VLA/OpenVLA/` | ✅ | 🔤 |
| 6 | Octo | 2405.12250 | `VLA/Octo/` | ✅ | 🔤 |
| 7 | π0 (Pi-Zero) | 2410.24164 | `VLA/Pi0/` | ✅ | 🔤 |
| 8 | RoboCat | 2306.11706 | `VLA/RoboCat/` | ✅ | 🔤 |
| 9 | BC-Z | 2203.02827 | `VLA/BC-Z/` | ✅ | 🔤 |
| 10 | VIMA | 2210.03094 | `VLA/VIMA/` | ✅ | 🔤 |
| 11 | What Matters in IL | 2106.00672 | `IL-Action-Head/IL-Analysis/` | ✅ | 🔤 |
| 12 | 3D Diffusion Policy | 2406.01586 | `IL-Action-Head/3D-Diffusion-Policy/` | ✅ | 🔤 |
| 13 | BridgeData V2 | 2309.12247 | `Datasets/BridgeData-V2/` | ✅ | 🔤 |
| 14 | DROID | 2303.03135 | `Datasets/DROID/` | ✅ | 🔤 |
| 15 | Flow Matching intro | 2404.08427 | `IL-Action-Head/Flow-Matching/` | ✅ | 🔤 |

**索引入口**：[论文索引 · 0 · VLA 算法层](../paper/论文索引.md#0-algorithm)

---

## 与现有论文索引的关系

| 现有索引章节 | 内容 | 本清单对应 |
|-------------|------|-----------|
| §0 综述 | 数据+全貌 | Layer 6 |
| §1 ALOHA | 数据+ACT | Layer 0–1 |
| §2 UMI | 数据 | 配合 Layer 1 Diffusion |
| §4 Foundation | RDT/Qwen | Layer 5 |
| §5 Ego | 人类视频 | Layer 7 |
| **§Algorithm** | RT/OpenVLA/π0 | Layer 2–5 · [论文索引 §0 算法层](../paper/论文索引.md#0-algorithm) |

---

> **下一步（你说论文下载放后面）**：你先按 Layer 0→6 顺序阅读；算法层 15 篇 PDF 已入库，翻译完成后索引会自动更新中文/对照链接。
