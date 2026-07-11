---
title: "RDT: Robotics Diffusion Transformer 系列"
authors: "Songming Liu, Bangguo Li, Kai Ma, Lingxuan Wu, et al. (清华大学)"
year: "2024 (RDT-1B) / 2026 (RDT2)"
source:
  - "RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation"
  - "RDT2: Exploring the Scaling Limit of UMI Data for Zero-Shot Cross-Embodiment Generalization"
tags:
  - RDT
  - RDT-1B
  - RDT2
  - foundation-model
  - diffusion-transformer
  - VLA
  - bimanual-manipulation
  - unified-action-space
  - UMI
  - pretrain-finetune
aliases:
  - Robotics Diffusion Transformer
  - RDT Foundation Models
paper_path:
  - "paper/Foundation-Models/RDT-1B/RDT-1B- a Diffusion Foundation Model for Bimanual Manipulation.pdf"
  - "paper/Foundation-Models/RDT2/RDT2.pdf"
project_website: "https://rdt-robotics.github.io"
related_notes:
  - "note/paper-note/ALOHA/ALOHA-Learning-Fine-Grained-Bimanual-Manipulation.md"
  - "note/paper-note/UMI-Universal-Manipulation-Interface.md"
  - "note/paper-note/ALOHA/ACT-Model-Working-Principles.md"
---

## 一句话总结

**RDT-1B** 用 **1.2B DiT 扩散模型 + 128 维物理可解释统一动作空间**，在 **46 个数据集 ~1M 轨迹** 上预训练、**6K+ 双臂轨迹** 微调，成为首个大规模双臂 diffusion FM，在 ALOHA 上比 ACT/OpenVLA/Octo **平均成功率 +56%**，具备未见物体/场景/指令的零样本与小样本能力。**RDT2** 则把路线推到 **UMI 10,000+ 小时人类数据 + 7B Qwen2.5-VL 三阶段训练（RVQ → Flow Matching → 单步蒸馏）**，成为首批在 **未见物体 + 场景 + 指令 + embodiment** 四因素上联合零样本泛化的 VLA 之一。

---

## 论文基本信息

| 项 | RDT-1B | RDT2 |
|----|--------|------|
| 标题 | A Diffusion Foundation Model for Bimanual Manipulation | Exploring the Scaling Limit of UMI Data for Zero-Shot Cross-Embodiment Generalization |
| 机构 | 清华大学 | 清华大学 |
| arXiv | 2024 | arXiv:2602.03310 (2026) |
| 核心规模 | **1.2B** DiT | **7B** Qwen2.5-VL + **400M** 动作专家 |
| 数据 | 预训练 46 数据集 **1M+** 轨迹 / 微调 **6K+** 双臂轨迹 | **10,000+ 小时** 增强 UMI（100 设备 × 100+ 家庭） |
| 部署平台 | ALOHA 双臂（ViperX） | 多种机械臂（UMI 同构夹爪 + 相机即可零样本迁移） |
| PDF | [论文索引 §45](../paper/论文索引.md) | [论文索引 §46](../paper/论文索引.md) |

---

## 在整条学习路线中的位置

```
ALOHA（低成本双臂 teleop + ACT）
    │
    ▼
UMI（无需机器人、野外人类演示）
    │
    ▼
RDT-1B（多机器人预训练 → 双臂微调，Unified Action Space）
    │
    ▼
RDT2（UMI 数据规模化 + VLM 三阶段，跨 embodiment 零样本）
```

| 对比维度 | ALOHA/ACT | RDT-1B | RDT2 |
|----------|-----------|--------|------|
| 模型规模 | ~80M CVAE | **1.2B** DiT | **7B** VLM + 400M 专家 |
| 数据来源 | 单机器人 teleop | 多机器人预训练 + 双臂微调 | **纯 UMI 人类数据**（无机器人 demo 预训练） |
| 动作建模 | CVAE 回归 chunk | **扩散** chunk | RVQ 离散 + **Flow Matching** 连续 |
| 跨机器人 | ❌ | 预训练共享，部署仍需目标机器人微调 | ✅ **零样本**跨 embodiment |
| 语言条件 | ACT 原版无 | ✅ T5-XXL | ✅ Qwen2.5-VL |

---

# Part A：RDT-1B

## 1. 研究动机

### 1.1 双臂操作的独特困难

与单臂相比，双臂操作有两个放大的难题：

1. **动作空间维度翻倍** → 数据更稀疏、分布更复杂
2. **多模态性更强** → 同一任务（如抓立方体）有多种合理模式（左抓 / 右抓 / 双手协作），人类 demo 随机选一种，确定性回归会学到「平均动作」而失败

### 1.2 数据与异构性的双重瓶颈

| 瓶颈 | 具体表现 |
|------|----------|
| **双臂数据极少** | 特定双臂机器人可用数据 **< 10K** 轨迹，不够训 FM |
| **多机器人异构** | 不同机器人 joint / EEF / gripper / base 格式完全不同 |
| **现有做法的局限** | 只挑动作空间相似的子集，或只保留公共维度 → **丢信息** |

### 1.3 论文要回答的核心问题

> 能否像 NLP 的 GPT 一样，用 **预训练 + 微调**，把单臂/多机器人数据「借」给双臂，训出一个可泛化的 diffusion 基础模型？

RDT 的答案：**可以** —— 关键是 **DiT 扩散架构** + **128 维 Unified Action Space** + **大规模预训练**。

---

## 2. 任务形式化

给定语言指令 ℓ，策略在时刻 t 接收观测 **o_t**，输出动作 **a_t**。

```
o_t := (X_{t-T_img+1:t+1}, z_t, c)
```

| 符号 | 含义 |
|------|------|
| **X** | RGB 图像历史，默认 **T_img = 2**（当前帧 + 前一帧） |
| **z_t** | 低维本体感觉（proprioception） |
| **c** | **控制频率**（Hz），不同机器人控制率不同，显式输入 |
| **a_t** | 通常是 **z_{t+1} 的子集**（目标下一时刻状态） |

**相机布局（3 路）**：

- 外部第三人称相机 **X1**
- 右手腕相机 **X2**
- 左手腕相机 **X3**

**动作块（Action Chunking）**：一次预测 **a_{t:t+T_a}**（与 ACT 相同思路），鼓励时序一致、减少 compounding error。

---

## 3. 整体架构

```
语言 ℓ ──→ T5-XXL（冻结）──┐
                           ├── 交叉注意力（交替注入）──→ DiT × L 层 ──→ MLP 解码器 ──→ 去噪动作块
3 路 RGB ──→ SigLIP（冻结）─┘         ↑
                                      │
本体 z_t、带噪动作 ã、频率 c、扩散步 k ──→ MLP + 傅里叶特征
```

### 3.1 为什么选扩散而不是回归 / 离散化

| 方法 | 问题 |
|------|------|
| 确定性回归 | 多模态 demo → 学到「平均模式」→ 分布外动作 |
| VAE（ACT） | 表现力不足 |
| 离散 token（OpenVLA） | 量化误差 |
| **扩散（RDT）** | 建模 **p(a\|ℓ, o)** 的多模态分布；动作维度低，采样开销可接受 |

### 3.2 针对机器人数据的 DiT 三项改造

| 改造 | 动机 | 效果 |
|------|------|------|
| **QKNorm + RMSNorm** | 机器人物理量数值范围不稳定；LayerNorm 居中破坏时序对称 | 否则 **1M 步预训练会发散** |
| **MLP 解码器**（替代线性头） | 机器人动作非线性强 | 否则灵巧任务（如推摇杆）失败 |
| **交替条件注入 ACI** | 图像 token 远多于文本 → 文本被压制 | 否则指令跟随差 |

### 3.3 训练时的输入掩码

每个多模态输入以 **10% 概率独立 mask**，防止模型只盯外部相机、忽略腕部深度信息。

---

## 4. 核心创新：128 维 Unified Action Space

这是 RDT-1B **最重要的工程贡献**，解决「如何在异构多机器人数据上预训练」。

### 4.1 设计原则

1. **物理可解释**：每个维度有明确物理含义（关节角、EEF 位姿、夹爪、基座速度…）
2. **统一格式**：不同机器人动作 **按物理含义填到对应槽位，其余补零**
3. **zt 与 at 同空间**：因为 **a_t ≈ z_{t+1} 的子集**，本体感觉空间自然包含动作空间

### 4.2 128 维向量结构（摘要）

| 索引范围 | 物理量 |
|----------|--------|
| [0, 10) | 右臂关节位置 |
| [10, 15) | 右夹爪 |
| [15, 25) | 右臂关节速度 |
| [25, 30) | 右夹爪速度 |
| [30, 33) | 右 EEF 位置 |
| [33, 39) | 右 EEF 6D 位姿 |
| [39, 42) | 右 EEF 速度 |
| [42, 45) | 右 EEF 角速度 |
| [45, 50) | 保留 |
| [50, 60) | **左臂**关节位置（对称） |
| … | 左臂 EEF、速度等同理 |
| [100, 102) | 基座线速度 |
| [102, 103) | 基座角速度 |
| [103, 128) | 保留 |

**映射规则**：

- 单臂机器人 → 映射到「右臂」槽位
- 6-DoF 臂 → 10 个关节槽位填前 6 个
- 7-DoF 臂（如 Franka）→ 填前 7 个
- EEF 旋转用 **6D 表示**（Zhou et al. 2019）

### 4.3 与 UMI / ALOHA 动作空间的关系

| 系统 | 原始动作 | 嵌入 Unified Space 后 |
|------|----------|----------------------|
| **ALOHA** | 14 维 joint + gripper ×2 | 填入左右臂 joint 槽位 |
| **UMI** | 6D EEF pose + gripper | 填入 EEF 6D + gripper 槽位 |
| **RT-1 类** | EEF + base + gripper | 填入对应 EEF + base 槽位 |

预训练时模型在 **128 维统一空间** 上学「物理规律」；微调到 ALOHA 时只激活相关维度。

---

## 5. 数据

### 5.1 预训练：46 数据集，1M+ 轨迹，21 TB

- 来源：RT-1、Bridge、DROID、Franka Kitchen 等 **Open X-Embodiment 生态**
- 采样权重：∝ √N_j，避免大集过采样、小集欠采样
- 训练中根据各集 loss 动态调权重

### 5.2 微调：自采 ALOHA 双臂数据集 6K+ 轨迹

| 维度 | 规模 |
|------|------|
| 任务数 | **300+**（抓取、插线、写公式…） |
| 物体 | **100+**（刚体/非刚体，多尺寸纹理） |
| 场景 | **15+** 房间，不同光照 |
| 指令 | 人工标注 + **GPT-4-Turbo 改写** 增多样性 |

**注意**：RDT-1B 的目标是 **用多机器人数据增强双臂泛化**，不是训「一个模型通吃所有机器人」—— 部署前仍需在目标双臂上微调。

---

## 6. 训练与推理

| 阶段 | 配置 |
|------|------|
| 预训练 | 48× H100 80GB，**1M 步**，约 1 个月 |
| 微调 | 同样 GPU，**130K 步**，约 3 天 |
| 推理加速 | DPM-Solver++：100 步 → **5 步** |
|  onboard 性能 | RTX 4090：**6 Hz** 动作块 / **381 Hz** 单步动作 |

**损失**：标准扩散去噪 MSE，对动作块 **a_{t:t+T_a}** 建模。

---

## 7. 实验（ALOHA 真实机器人）

### 7.1 七项挑战性任务

| 任务 | 考察能力 | 训练 demo 量 |
|------|----------|-------------|
| 洗杯子 | 未见物体 | 133（仅见过杯） |
| 倒水 | 未见场景（3 个新房间） | 350（仅见过房间） |
| 倒水-左-1/3 / 右-2/3 | **未见指令**（「三分之一」「三分之二」） | 各 18+19+19 |
| 交接 | **5-shot** 新技能 | 5 |
| 折叠短裤 | **1-shot** 新技能 | 1 |
| 机器狗 | **灵巧操作**（推摇杆角度） | 68 |

### 7.2 主要结果（表 3 摘要）

| 任务 | ACT | OpenVLA | Octo | RDT（无预训练） | **RDT（预训练）** |
|------|-----|---------|------|-----------------|-------------------|
| 洗杯（未见杯2） | 0% | 0% | 0% | 12.5% | **87.5%** |
| 倒水（未见房间） | 0% | 0% | 0% | 75% | **100%** |
| 倒水-左-1/3 | — | 0% | 12.5% | 62.5% | **100%** |
| 交接（5 demo） | 0% | 0% | 0% | 88% | **100%** |
| 折叠短裤（1 demo） | 0% | 0% | 0% | 75% | **100%** |
| 机器狗 | 32% | 0% | 4% | 64% | **76%** |

**关键结论**：

- **Q1/Q2**：未见物体、场景、指令（如「三分之一水」）均可零样本
- **Q3**：5-shot / 1-shot 学全新技能（交接、折衣服）
- **Q4**：毫米级灵巧（推摇杆控机器狗直线走）
- **Q5 消融**：预训练 + 扩散 + 大模型 **缺一不可**；回归版 RDT 显著差于扩散版

---

# Part B：RDT2

## 8. RDT2 动机：RDT-1B 之后还缺什么

RDT-1B 解决了「双臂 + 多机器人预训练」，但仍有两个痛点：

| 痛点 | 说明 |
|------|------|
| **仍依赖目标机器人微调** | 新 embodiment 上要采数百小时数据 |
| **VLA 架构效率** | 纯扩散 VLA 收敛慢；纯离散 VLA 有量化误差；大模型推理慢 |

RDT2 的策略：**把 UMI 数据推到极限 + 用 VLM 三阶段对齐离散语言知识与连续控制**。

---

## 9. 数据：增强 UMI + 10,000+ 小时

### 9.1 硬件重设计（相对原始 UMI）

| 项目 | 原始 UMI | RDT2 UMI |
|------|----------|----------|
| 制造 | 3D 打印 PLA/PETG | **CNC 尼龙66 + 玻璃纤维** |
| 追踪 | SLAM | **红外光追踪**（高速/无纹理/透明背景更稳） |
| 夹爪 | 平行夹爪 | **连杆夹爪**（窄缝/杂乱环境更好） |

**部署条件**：相同型号 **RGB 相机 + 连杆夹爪** 装到任意机械臂 → 策略零样本迁移。

### 9.2 数据规模

- **10,000+ 小时** 真实家庭环境操作
- **100 台** 设备 × **100+** 家庭
- **零机器人 demo** 参与预训练（纯人类 UMI + 少量 VLM 图文对）

### 9.3 Data Pyramid（数据金字塔）

```
        塔尖：Teleop 机器人数据（高保真、最贵、实验室）
          │
        中层：仿真数据（便宜、sim-to-real gap）
          │
        基底：互联网视频 / UMI 人类数据（量大、缺动作标签 → UMI 补动作）
```

RDT2 论证：**可扩展的数据采集方式（UMI）+ 规模化** 是 VLA 泛化的关键。

---

## 10. 三阶段训练流水线

```
Stage 1: RVQ 离散化 + 7B Qwen2.5-VL 交叉熵预训练（128K iter）
              │
              ▼
Stage 2: 冻结 VLM，400M Flow Matching 动作专家（66K iter，5 积分步）
              │
              ▼
Stage 3: 扩散蒸馏 → 单步生成器 RDT2-UltraFast（实时推理）
```

### Stage 1：RVQ + VLM

- 连续动作块 **A_t ∈ R^{T_a × d}** → 1D CNN 编码 → **RVQ 量化** → 离散 token
- 占用 Qwen 词表中 **1024 个最少用 token**
- **交叉熵下一 token 预测**，与 VLM 原生训练目标一致 → **不破坏预训练离散知识**
- 混合数据：UMI + 少量视觉-语言对

### Stage 2：Flow Matching 动作专家

- **400M** 参数（RDT-1B 架构变体，GQA 加速）
- **冻结** Stage 1 的 VLM；专家用 **交叉注意力** 融合 VLM 各层特征
- **Flow Matching** 损失（比标准扩散收敛更快）
- 推理：**5 步**积分（τ=0.2）

**两个变体**：

| 变体 | 动作生成 |
|------|----------|
| **RDT2-VQ** | Stage 1 纯自回归离散 token |
| **RDT2-FM** | Stage 1 + Stage 2 连续 flow matching |

### Stage 3：单步蒸馏（UltraFast）

- 目标：高动态任务（**打乒乓球**）需要极低延迟
- 把 5 步专家 **蒸馏成 1 步**生成器
- **RDT2-UltraFast**：比 π0.5（3B）还快，尽管总参数 ~7B+

| 模型 | 推理频率（约） |
|------|----------------|
| π0-FAST (3B) | 2.7 Hz |
| π0.5 (3B) | 1.6 Hz |
| RDT2-VQ (7B) | 16.0 Hz |
| RDT2-FM (7B) | 17.0 Hz |
| **RDT2-UltraFast (7B)** | **23.0 Hz** |

---

## 11. RDT2 实验

### 11.1 零样本实验（4U：Unseen 四因素）

**不做任何微调**，部署在 **未见 embodiment** 上。

| 任务类型 | 示例 |
|----------|------|
| Pick | 开放词汇抓取 |
| Pick & Place | 指定物体放到指定位置 |
| Wipe | 任意布擦桌子 |
| Press | 按任意按钮 |
| Shake | 摇晃物体 |

**RDT2-FM 平均成功率 ~44%**（256 次试验/任务）；**RDT2-VQ ~41%**。

意义：仅人类 UMI 数据 + 大 VLM，首次在 **物体 + 场景 + 指令 + embodiment** 上同时零样本 —— 成功率不高，但组合泛化Previously 很难。

### 11.2 缩放定律

同时增大 **模型参数 N** 和 **数据 token D** → 训练 loss 可预测下降（Chinchilla 型规律）。说明 UMI 路线 **还能继续 scale**。

### 11.3 微调 vs π0 系列

在 **可变形物体、长时域、高动态** 任务（折衣服、拉拉链、打乒乓球、快速按钮）上，**RDT2-UltraFast** 优于 **π0-FAST** 和 **π0.5**。

---

## 12. RDT-1B vs RDT2 vs 其他 FM

| 维度 | OpenVLA | Octo | π0 | RDT-1B | RDT2 |
|------|---------|------|-----|--------|------|
| 参数量 | 7B | 93M | 3B | **1.2B** | **7B** + 400M |
| 动作表示 | 离散 token | 扩散 | 扩散 | 扩散 chunk | RVQ + Flow |
| 预训练数据 | 多机器人 | 多机器人 | 多机器人 | 46 集 1M | **UMI 10k hr** |
| 目标机器人 | 需微调 | 需微调 | 需微调 | **ALOHA 微调** | **零样本跨 embodiment** |
| 双臂专精 | 通用 | 通用 | 通用 | ✅ **核心场景** | 通用 + UMI 迁移 |
| 推理速度 | 慢（自回归） | 中 | 中 | 6 Hz chunk | **23 Hz** UltraFast |

---

## 13. 读论文时建议关注的实现细节

### RDT-1B

1. **Unified Action Space 怎么填零** — 附录 C 表 4，对接 Open X 数据预处理
2. **ACI 交替注入** — 图像/语言条件如何进 DiT
3. **控制频率 c** — 不同数据集 Hz 不同，作为显式输入
4. **微调数据构成** — 300+ 任务如何覆盖泛化维度

### RDT2

1. **RVQ 码本设计** — 防 codebook collapse 的四项措施
2. **Stage 1→2 为何冻结 VLM** — 保留离散预训练知识
3. **UMI 硬件 diff** — 从 SLAM 到 IR tracking 的原因
4. **零样本实验 protocol** — 256 次试验、去重指令、新物体新场景

---

## 14. 局限与后续方向（论文隐含）

| 局限 | 说明 |
|------|------|
| RDT-1B 非 cross-embodiment | 预训练用统一空间，但部署 ALOHA 前 **必须微调** |
| RDT2 零样本成功率仍有限 | ~40–44%，离实用部署有距离 |
| 依赖 UMI 硬件同构 | 相机 + 连杆夹爪型号需一致 |
| 算力 | RDT-1B 预训练 48×H100×1月；RDT2 7B 更重 |

---

## 15. 推荐阅读顺序（不限论文）

```
① 视频（建立直觉，~76 min）
   B站「RDT-1B：解密全球最大的双臂机器人扩散大模型」
   → 机器之心 × 刘松铭（清华 TSAIL），口述动机 / Unified Action Space / demo
   → https://www.bilibili.com/video/BV1FjyHYmEDQ

② 项目页 demo（5 min）
   → https://rdt-robotics.github.io/

③ 本笔记（结构化速读，30–45 min）

④ 本地 RDT-1B 中文 PDF — Method + §4.2 Unified Action Space

⑤ RDT2：先读 [UMI 笔记](./UMI-Universal-Manipulation-Interface.md)，再读 RDT2 PDF
```

更多视频、博客、代码入口 → [学习资料全景索引](../resources/links/资源索引.md#1-rdt--双臂-diffusion-fm)

---

## 16. 相关资源

| 类型 | 链接 |
|------|------|
| **视频（首选入门）** | [Bilibili：RDT-1B 解密…](https://www.bilibili.com/video/BV1FjyHYmEDQ) · 机器之心 · ~76 min |
| 项目页 + demo 视频 | [rdt-robotics.github.io](https://rdt-robotics.github.io/) |
| 代码 + 部署 | [github.com/thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) |
| 权重 | [HuggingFace rdt-1b](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) |
| 论文索引 | [paper/论文索引.md](../paper/论文索引.md) §45–46 |
| 多媒体资料总表 | [学习资料全景索引](../resources/links/资源索引.md) |
| ALOHA 笔记（baseline 机器人） | [ALOHA 笔记集](./ALOHA/概述.md) |
| UMI 笔记（RDT2 数据来源） | [UMI-Universal-Manipulation-Interface.md](./UMI-Universal-Manipulation-Interface.md) |
| ACT 原理 | [ACT-Model-Working-Principles.md](./ALOHA/ACT-Model-Working-Principles.md) |
| 整体认知地图 | [VLA与机器人整体认知地图.md](../VLA与机器人整体认知地图.md) |
| 学习路线 | [VLA算法层学习路线与论文清单.md](../VLA算法层学习路线与论文清单.md) |

---

## 17. 自测清单（读完应能回答）

- [ ] RDT 为什么用扩散而不是 ACT 式 CVAE？
- [ ] 128 维 Unified Action Space 如何容纳 ALOHA joint 和 UMI EEF？
- [ ] 预训练 46 数据集 vs 微调 6K 双臂数据各自解决什么问题？
- [ ] ACI、QKNorm、MLP 解码器分别解决什么训练问题？
- [ ] RDT2 三阶段各自训练什么、冻结什么？
- [ ] RDT2 如何实现「未见 embodiment」零样本？硬件前提是什么？
- [ ] RDT-1B 和 RDT2 的数据路线根本差异是什么？
