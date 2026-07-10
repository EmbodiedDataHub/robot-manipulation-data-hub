---
title: "零基础 · RDT 向多模态 FM 学习路线"
audience: "具身智能小白，计划做「多源数据统一 + Foundation Model 预训练」"
goal: "类似 RDT：Unified Action Space + 大规模 Pretrain + 目标机器人 Finetune"
duration: "约 12–16 周（每天 1–2 小时）"
updated: "2026-07-08"
related:
  - "快速入门-最短学习路径.md"
  - "VLA与机器人整体认知地图.md"
  - "VLA训练与数据全貌-深度版.md"
  - "paper-note/RDT-Foundation-Models.md"
  - "resources/links/资源索引.md"
  - "../paper/论文索引.md"
---

# 零基础 · RDT 向多模态 FM 学习路线

> **你未来的工作，拆成两句话就是：**  
> ① 把来自不同机器人、不同采集方式的数据，**翻译成同一套「物理语言」**（Unified Action / Observation）；  
> ② 在这套统一格式上 **Pretrain 一个 Foundation Model**，再在目标场景 **Finetune**。

本路线按这个终点倒推，从零基础到能独立设计数据协议 + 读懂 RDT 级论文。**不假设你会机器人学或扩散模型。**

---

## 一、先搞懂：RDT 式 FM 到底在做什么

```
┌─────────────────────────────────────────────────────────────────┐
│  你们要做的事（RDT 已验证过的 recipe）                              │
├─────────────────────────────────────────────────────────────────┤
│  数据侧                         │  模型侧                         │
│  · 多源：Teleop / UMI / 开源集   │  · VLM 编码 图像 + 语言（常冻结） │
│  · 统一 Action Space（128 维）   │  · DiT / Flow 扩散 Action Head │
│  · RLDS / LeRobot 格式 + QA     │  · Pretrain 1M+ 步              │
│  · 混合采样权重 + 语言重标注      │  · Finetune 目标机器人          │
└─────────────────────────────────────────────────────────────────┘
```

| 模块 | RDT-1B 做法 | 你们需要决策的点 |
|------|-------------|------------------|
| **Action 统一** | 128 维物理可解释向量，按含义填槽、其余补零 | 维度设计、单臂/双臂/EEF/joint 如何映射 |
| **Obs 统一** | 3 路 RGB + proprio + 控制频率 c | 相机数量、是否用 depth、历史帧数 T_img |
| **Pretrain 数据** | 46 数据集 ~1M 轨迹 | 用 OXE 子集还是自采 + 开源混合 |
| **Finetune 数据** | 6K+ 自采双臂（ALOHA） | 目标 embodiment 高质量 demo |
| **Action Head** | Diffusion Transformer 1.2B | DiT vs Flow vs 离散 token（RDT2 混合） |
| **底座** | SigLIP + T5-XXL（冻结） | SigLIP / Qwen-VL / PaliGemma |

**RDT2 的延伸**（若你们还要 **跨 embodiment 零样本**）：数据侧主攻 **UMI 规模化**，模型侧 **RVQ + VLM + Flow Matching**。这是 Phase 5 再深入的内容。

---

## 二、学习地图总览（5 个 Phase）

```
Phase 0  建立直觉（1 周）     机器人/VLA 是什么 · 模仿学习 · 四种 action 写法
    ↓
Phase 1  单任务 Policy（2 周） ACT · Diffusion Policy · 一条数据长什么样
    ↓
Phase 2  数据采集与异构（3 周） ALOHA · UMI · OXE · 五条数据产线
    ↓
Phase 3  数据统一与混训（3 周）★ Unified Action · 八步流水线 · 负迁移
    ↓
Phase 4  VLA / FM 架构（3 周）  RT-2 · OpenVLA · Octo · π0 · Recipe 对照
    ↓
Phase 5  RDT 主线 + 落地（4 周）★ RDT-1B/RDT2 精读 · 代码 · 最小复现计划
```

**★ = 和你们项目直接相关，时间不够时 Phase 0→1→3→5 不可省，Phase 2/4 可压缩。**

**推荐总时长**：全职学习 **8–10 周**；业余 **12–16 周**（每天 1–2 小时）。

---

## 三、Phase 0 · 建立直觉（第 1 周）

### 本阶段目标

- 能向别人解释：VLA、模仿学习、Teleop、Action Chunk 是什么
- 知道 joint / EEF / delta / unified 四种 action 写法的区别
- **不要求**推公式

### 学习顺序

| 天 | 做什么 | 资料 | 时长 |
|:--:|--------|------|:----:|
| D1 | 全局地图 | [VLA 与机器人整体认知地图](./VLA与机器人整体认知地图.md) §一–§三 | 1.5h |
| D2 | 模仿学习 | [快速入门 Day 1–2](./快速入门-最短学习路径.md) + [LeRobot 中文教程](https://hugging-face.cn/docs/lerobot/il_robots) | 1.5h |
| D3 | 四种 action | [快速入门 Day 3](./快速入门-最短学习路径.md) + [训练全貌 §4.2](./VLA训练与数据全貌-深度版.md) | 1h |
| D4 | Diffusion 直觉 | [Diffusion Policy 中文博客](https://deathgarurumon.github.io/docs/diffusion_policy/) + [官网 demo](https://diffusion-policy.cs.columbia.edu/) | 1h |
| D5 | 看视频建立感性认识 | [ALOHA 官网](https://tonyzhaozh.github.io/aloha) + [UMI 官网](https://umi-gripper.github.io) 各 10 min | 0.5h |
| D6–D7 | 自测 + 画一张图 | 见下文 §十 Phase 0 自测 | 1h |

### 本阶段不读论文

先把名词搞对，避免一上来被 Unified Action Space 劝退。

---

## 四、Phase 1 · 单任务 Policy 基础（第 2–3 周）

### 本阶段目标

- 理解 **BC**：输入 (图像, 语言) → 输出 action / action chunk
- 理解 **为什么** 要 Chunking 和 Diffusion（多模态 demo、误差累积）
- 能描述 **一条训练样本** 有哪些字段

### 论文阅读顺序（Phase 1）

| 顺序 | # | 论文 | 本地 | 读什么 | 搞懂什么 |
|:----:|:-:|------|:----:|--------|---------|
| 1 | 26 | **ALOHA** | ✅🔤 | Intro + Method + ACT 段 | 双臂 teleop、joint space、Action Chunking |
| 2 | 12 | **Diffusion Policy** | ✅🔤 | §1–3 + Method 主干 | 从噪声 denoise 出动作轨迹、horizon H |
| 3 | — | — | — | [ACT 原理笔记](./paper-note/ALOHA/ACT-Model-Working-Principles.md) | CVAE、Temporal Ensembling（笔记替代啃全文） |
| 4 | 11 | What Matters in IL | ✅🔤 | Abstract + 结论 | 数据量/质量/增强的经验规律（选读） |

### 配合资料

| 类型 | 链接 |
|------|------|
| **视频（ACT 首选）** | [Bilibili：ACT / ALOHA 精析](https://www.bilibili.com/video/BV1xGF3eeEjB) · ~42 min |
| 视频 | [ALOHA demo](https://tonyzhaozh.github.io/aloha) |
| 博客 | [Diffusion Policy 精读](https://devon018.github.io/Diffusion_Policy/)（可选） |
| 动手 | LeRobot 跑通 ACT 或读透 `--policy=act` 配置（有 GPU 再做） |

### Phase 1 过关标准

- [ ] 能画出：Teleop → HDF5/LeRobot → BC 训练 → 部署 的闭环
- [ ] 能解释 Diffusion Policy 和 ACT 各适合什么类型的 demo
- [ ] 能写出一条 step 的 `observation` / `action` 字段名

---

## 五、Phase 2 · 数据采集与多源异构（第 4–6 周）

### 本阶段目标

- 知道 **五条数据产线**（Teleop / UMI / Sim / Ego / 混合）的 fidelity vs scale trade-off
- 理解 **为什么 FM 需要多源数据**，以及 **异构** 体现在哪（action 维度、Hz、相机、语言）
- 为 Phase 3「统一格式」积累直觉

### 论文阅读顺序（Phase 2）

| 顺序 | # | 论文 | 本地 | 读什么 | 和 FM 的关系 |
|:----:|:-:|------|:----:|--------|-------------|
| 1 | 1 | **VLA 数据 / Data Engine 综述** | ✅🔤 | §1–3 + §5 | 数据-centric 全局框架 |
| 2 | 2 | **Open X-Embodiment** | ✅🔤 | Abstract + RLDS + 混合动机 | 跨 embodiment 预训练的「行业标准」 |
| 3 | 34 | **UMI** | ✅🔤 | System + Policy Interface | 无机器人采数据、relative action |
| 4 | — | — | — | [UMI 笔记](./paper-note/UMI-Universal-Manipulation-Interface.md) | SLAM、延迟对齐、Diffusion 搭配 |
| 5 | 24 | **DROID** | ✅🔤 | Abstract + 数据协议 | in-the-wild teleop 大规模范例 |
| 6 | 23 | BridgeData V2 | ✅🔤 | 选读 Dataset 节 | 低成本桌面 setup 参考 |
| 7 | 4 | **MimicGen** | ✅🔤 | Abstract | 仿真扩增思路（了解即可） |
| 8 | 26→28 | Mobile ALOHA | ✅🔤 | 选读 | 移动 + 双臂数据形态 |

**数据采集工程（不读论文也要看博客）**：

- [RAI 五类数据源对比](https://rai-inst.com/resources/blog/handheld-robotic-data-collection/)
- [SVRC 数据管线 6 阶段](https://www.roboticscenter.ai/learn/robotics-library/data-collection-pipeline-overview)
- [训练全貌 §五–§六](./VLA训练与数据全貌-深度版.md)

### Phase 2 过关标准

- [ ] 能对比 ALOHA vs UMI vs DROID 的 action 空间、采集成本、适用场景
- [ ] 能解释 RLDS / LeRobot 各自适合什么训练栈
- [ ] 能说出「把 3 个开源数据集混一起训」会遇到哪 3 类坑（维度、Hz、语言）

---

## 六、Phase 3 · 数据统一与混训（第 7–9 周）★ 核心

### 本阶段目标

- **设计思维**：如何定义 Unified Action Space（物理含义 > 数学 trick）
- 掌握 Raw → Trainable **八步流水线**
- 理解 Pretrain vs Finetune **数据策略**差异、负迁移与 Data Pyramid

### 必读材料（顺序）

| 顺序 | 材料 | 读什么 |
|:----:|------|--------|
| 1 | [训练全貌 深度版](./VLA训练与数据全貌-深度版.md) | **全文**，重点 §二 Stage 1/2、§六八步流水线、§七 Recipe、§九 Data Pyramid |
| 2 | [VLA 数据综述笔记](./paper-note/VLA-Datasets-Benchmarks-Data-Engines.md) | Unified Data Engine、Video-to-Data（扩展数据源时用） |
| 3 | [数据调研报告](./机器人数据工作综合调研报告.md) | § Unified Action、§ 负迁移、§ 2026 共识 |
| 4 | #48 **ABot-M0** | 数据清洗 → UniACT 600 万轨迹管线（工程参考） |
| 5 | #47 **Qwen-RobotManip** | Alignment-first：表示/运动/行为三维对齐（与 RDT 对照） |

### 论文阅读顺序（Phase 3 · 数据向）

| 顺序 | # | 论文 | 重点章节 |
|:----:|:-:|------|---------|
| 1 | 2 | Open X-Embodiment | 数据格式、 embodiment 标签、混合训练 |
| 2 | 48 | ABot-M0 | 异构 raw → 统一表示的 pipeline |
| 3 | 47 | Qwen-RobotManip | Camera-centric delta、co-training 数据配比 |
| 4 | 5 | Robo-DM（选读） | 大规模存储/加载/压缩 |

### 本阶段你要输出的「项目文档 0.1」（建议自己写一页纸）

```
1. 我们支持哪些 embodiment？（单臂/双臂/移动底）
2. Unified Action 向量每一维的物理含义（参考 RDT 128 维表）
3. 每种数据源 → 统一空间的映射规则（填哪、补零哪）
4. 控制频率 c 怎么处理（显式输入 vs 重采样到统一 Hz）
5. 图像：几路相机、分辨率、是否 mask 某路
6. 语言：模板 / 人工 / VLM 生成 / GPT 改写
7. Pretrain 混合权重（√N 采样？按 loss 动态调？）
8. Finetune 数据量与任务覆盖计划
```

### Phase 3 过关标准

- [ ] 能对照 RDT 表 4，解释 ALOHA joint 和 UMI EEF 如何填进 128 维
- [ ] 能画出八步数据处理流水线并说出每步输入输出
- [ ] 能解释 Pretrain「要广」、Finetune「要深」

---

## 七、Phase 4 · VLA / FM 架构（第 10–12 周）

### 本阶段目标

- 理解 VLA **四模块**：Vision Encoder + Language + Fusion + Action Head
- 对比三种 Action Head：**离散 token / Diffusion / Flow Matching**
- 知道 OpenVLA、Octo、π0 的 recipe，避免只盯 RDT 一家

### 论文阅读顺序（Phase 4 · 模型向）

| 顺序 | # | 论文 | 本地 | 读什么 | 为何读 |
|:----:|:-:|------|:----:|--------|--------|
| 1 | 13 | **RT-1** | ✅🔤 | Method | 大规模 multi-task Transformer BC 鼻祖 |
| 2 | 14 | **RT-2** | ✅🔤 | Method + co-train | VLM + action token，VLA 范式起源 |
| 3 | 18 | **OpenVLA** | ✅🔤 | Architecture + Training | 开源 7B、离散 action、OXE 970K |
| 4 | 19 | **Octo** | ✅🔤 | Method | 小扩散 FM、模块化 finetune |
| 5 | 21 | **π0** | ✅🔤 | Method | Flow Matching + PaliGemma，工业级对照 |
| 6 | 25 | Flow Matching 教程 | ✅🔤 | §1–2 | π0/RDT2 的理论基础（选读） |
| 7 | 6 | VLA Anatomy 综述 | ✅🔤 | 模块划分 | 串联全局（选读） |

### 配合资料

| 类型 | 链接 |
|------|------|
| 博客 | [π0 首发文](https://www.pi.website/blog/pi0) · [DeepMind OXE](https://deepmind.google/blog/scaling-up-learning-across-many-different-robot-types/) |
| 对照 | [训练全貌 §七 Recipe 对照](./VLA训练与数据全貌-深度版.md) |
| 清单 | [VLA 算法层清单](./VLA算法层学习路线与论文清单.md) Layer 2–5 |

### Phase 4 过关标准

- [ ] 能填表对比 RT-2 / OpenVLA / π0 / RDT 的 Action Head、数据量、action 表示
- [ ] 能解释「冻结 VLM vs co-train」的 trade-off
- [ ] 能说出为什么 RDT 选 DiT 扩散而不是 OpenVLA 式离散 token

---

## 八、Phase 5 · RDT 主线精读 + 落地（第 13–16 周）★ 核心

### 本阶段目标

- **精读 RDT-1B**：Unified Action + DiT + 预训练/微调全流程
- **理解 RDT2**：UMI 10k 小时 + 三阶段 VLM 训练（若做跨 embodiment）
- 对照 **官方代码**，列出你们 MVP 与 RDT 的差异项

### 推荐学习顺序（视频优先）

```
① B站 [RDT-1B 解密…](https://www.bilibili.com/video/BV1FjyHYmEDQ)（~76 min）
② [RDT 官网 demo](https://rdt-robotics.github.io/)（5 min）
③ [RDT 解读笔记](./paper-note/RDT-Foundation-Models.md)（45 min）
④ 本地 RDT-1B 中文 PDF — §3 框架 · §4 Unified Action · §5 实验
⑤ 本地 RDT2 中文 PDF — 数据 + 三阶段（若需要 UMI 路线）
⑥ GitHub: thu-ml/RoboticsDiffusionTransformer — models/ + train/ + scripts/
```

### 论文阅读顺序（Phase 5）

| 顺序 | # | 论文 | 精读章节 | 产出 |
|:----:|:-:|------|---------|------|
| 1 | 45 | **RDT-1B** | §3–§5 + 附录 C（128 维表）+ 附录 D（预训练集） | Unified Action 设计文档 v1 |
| 2 | 46 | **RDT2** | §4 硬件 · §5 三阶段 · §6 零样本实验 | 数据规模化 / VLM 路线决策 |
| 3 | 45 | RDT-1B | 附录 F/H（训练超参、消融） | Pretrain recipe 清单 |
| 4 | 47 | Qwen-RobotManip | 与 RDT 对比章节 | 团队技术选型讨论材料 |
| 5 | 21 | π0 | 全文 skim | Action Head 备选方案 |

### 最小复现 / MVP 路线图（建议团队讨论）

| 阶段 | 目标 | 规模建议 | 验证指标 |
|------|------|---------|---------|
| **M0** | 单一数据源 + 小 DiT/MLP BC | 1 数据集、1 机器人、1K ep | 单任务成功率 > 随机 |
| **M1** | Unified Action 映射 + 2 源混合 | 2 种 action 格式 | 混合训练 loss 收敛、无 NaN |
| **M2** | + 语言条件 + 冻结 SigLIP/T5 | 加指令标注 | 简单 language 跟随 |
| **M3** | Pretrain 规模扩大 | 100K+ ep 或多数据集 | 零样本物体/场景小幅提升 |
| **M4** | 目标机器人 Finetune | 自采 500–6K ep | 接近 RDT 论文级需更多 |

**算力现实**：RDT-1B 全量预训练 = 48×H100×1 月。小白团队通常从 **RDT-170M  ablation 规模** 或 **Octo 93M** 量级起步，先验证 **数据统一 pipeline** 再 scale。

### Phase 5 过关标准

- [ ] 能白板讲清 RDT 输入输出、128 维空间、Pretrain/Finetune 分工
- [ ] 能列出 GitHub 里「改 unified action 映射」要动哪些文件
- [ ] 团队有一份 Unified Action Space v0.1 设计 doc

---

## 九、论文阅读顺序总表（按优先级一条线读下来）

> 编号与 [`paper/论文索引.md`](../paper/论文索引.md) 一致。✅ = 本地已有 · 🔤 = 有中文翻译。

### Tier S · 必读（做 RDT 向 FM 不可跳过）

| 总序 | # | 论文 | Phase | 笔记/视频 |
|:----:|:-:|------|:-----:|----------|
| 1 | — | （无论文）[VLA 认知地图](./VLA与机器人整体认知地图.md) | 0 | — |
| 2 | — | [训练全貌 深度版](./VLA训练与数据全貌-深度版.md) | 3 | — |
| 3 | 26 | ALOHA | 1 | [ACT/ALOHA B站](https://www.bilibili.com/video/BV1xGF3eeEjB) · [ALOHA 笔记集](./paper-note/ALOHA/概述.md) |
| 4 | 12 | Diffusion Policy | 1 | [博客](https://deathgarurumon.github.io/docs/diffusion_policy/) |
| 5 | 1 | VLA Data Engine 综述 | 2 | [数据笔记](./paper-note/VLA-Datasets-Benchmarks-Data-Engines.md) |
| 6 | 2 | Open X-Embodiment | 2–3 | — |
| 7 | 34 | UMI | 2 | [UMI 笔记](./paper-note/UMI-Universal-Manipulation-Interface.md) |
| 8 | 14 | RT-2 | 4 | — |
| 9 | 18 | OpenVLA | 4 | — |
| 10 | 19 | Octo | 4 | — |
| 11 | 21 | π0 | 4 | [π0 博客](https://www.pi.website/blog/pi0) |
| 12 | 45 | **RDT-1B** | 5 | [B站](https://www.bilibili.com/video/BV1FjyHYmEDQ) · [RDT 笔记](./paper-note/RDT-Foundation-Models.md) |
| 13 | 46 | **RDT2** | 5 | 同上 |

### Tier A · 强推荐（数据统一 / 工程）

| 总序 | # | 论文 | 何时读 |
|:----:|:-:|------|--------|
| 14 | 48 | ABot-M0 | Phase 3 数据管线设计时 |
| 15 | 47 | Qwen-RobotManip | Phase 3–5 技术选型对照 |
| 16 | 24 | DROID | Phase 2 大规模 teleop 参考 |
| 17 | 13 | RT-1 | Phase 4 补历史脉络 |
| 18 | — | [数据调研报告](./机器人数据工作综合调研报告.md) | Phase 3 |

### Tier B · 按需（扩展数据源 / 进阶）

| # | 论文 | 何时读 |
|:-:|------|--------|
| 4 | MimicGen | 要做仿真扩增 |
| 23 | BridgeData V2 | 低成本桌面 benchmark |
| 25 | Flow Matching 教程 | 选 Flow 而非 DDPM 时 |
| 28 | Mobile ALOHA | 移动 manipulation |
| 41+ | Ego / EgoVLA 等 | Phase 5 之后加人类视频预训练 |
| 6–10 | 各类 VLA 综述 | 写 related work 时 |

---

## 十、12 周业余学习计划（可打印）

| 周 | Phase | 核心任务 | 论文（#） | 视频/博客 |
|:--:|:-----:|---------|----------|----------|
| W1 | 0 | 认知地图 + 快速入门 Day 1–4 | — | LeRobot 中文 · Diffusion 博客 |
| W2 | 1 | ALOHA + ACT 笔记 | 26 | [ACT/ALOHA B站精析](https://www.bilibili.com/video/BV1xGF3eeEjB) · ALOHA 官网 |
| W3 | 1 | Diffusion Policy | 12 | DP 官网 demo |
| W4 | 2 | 数据综述 + OXE | 1, 2 | RAI 博客 |
| W5 | 2 | UMI + 笔记 | 34 | UMI 官网 |
| W6 | 2 | DROID + 数据产线 | 24 | SVRC 管线博客 |
| W7 | 3 | 训练全貌精读 | — | — |
| W8 | 3 | ABot-M0 + Qwen-RobotManip | 48, 47 | Label Studio meta |
| W9 | 3 | 写 Unified Action v0.1 | — | — |
| W10 | 4 | RT-2 + OpenVLA | 14, 18 | DeepMind OXE 博客 |
| W11 | 4 | Octo + π0 | 19, 21 | π0 博客 |
| W12 | 5 | RDT 视频 + 笔记 + PDF | 45 | **B站 RDT 76min** |
| W13–16 | 5 | RDT2 + 代码 + MVP 规划 | 46 | RDT GitHub |

---

## 十一、团队分工建议（若多人）

| 角色 | Phase 1–3 重点 | Phase 4–5 重点 |
|------|---------------|---------------|
| **数据** | 产线选型、RLDS/LeRobot、QA、Unified Action 映射实现 | Pretrain 混合权重、Finetune 采集计划 |
| **模型** | BC/Diffusion 基础、一条样本 tensor 形态 | DiT 结构、VLM 冻结策略、采样加速 |
| **系统** | 时间同步、存储格式、训练 dataloader | 4090 级推理 Hz、部署 pipeline |
| **算法（你）** | 通读全路线、做技术选型、写 protocol doc | 复现 RDT ablation、定 MVP 里程碑 |

**小白个人**：前 6 周 **数据+直觉** 和 **模型** 交替读（本路线已交错安排），不要只啃模型不看数据——**RDT 的壁垒一半在 Unified Action + 混合数据**。

---

## 十二、自测清单（开始自研前必须全 ✅）

### 概念

- [ ] VLA = VLM + Action Head；FM Pretrain = 多源机器人数据 + 大模型
- [ ] Unified Action Space 解决的是 **异构**，不是 **跨 embodiment 零样本**（后者 RDT2 + UMI 另说）
- [ ] Pretrain 广、Finetune 深；负迁移来自 action/obs 不对齐

### 数据

- [ ] 能设计 128 维（或你们自定 N 维）向量的物理含义表
- [ ] 能说出从 raw bag 到 training batch 的 8 步
- [ ] 知道语言标注至少 2 种做法及 QA 要点

### 模型

- [ ] 能解释 RDT 为何用扩散、DiT 三项改造（QKNorm / MLP 解码 / ACI）各解决什么
- [ ] 能对比 OpenVLA 离散 token vs RDT 连续 diffusion 的优劣
- [ ] 知道推理时 DPM-Solver++ 5 步、6 Hz chunk 的含义

### 工程

- [ ] 看过 RDT GitHub 目录结构
- [ ] 知道 LeRobot vs RLDS 选型对训练栈的影响
- [ ] 团队有 MVP 里程碑 M0–M2 时间表

---

## 十三、资料入口速查

| 需求 | 去哪里 |
|------|--------|
| 7 天极速入门 | [快速入门-最短学习路径](./快速入门-最短学习路径.md) |
| 全局导航 | [VLA 认知地图](./VLA与机器人整体认知地图.md) |
| 训练+数据深度 | [训练全貌 深度版](./VLA训练与数据全貌-深度版.md) |
| RDT 专题 | [RDT 笔记](./paper-note/RDT-Foundation-Models.md) + [B站视频](https://www.bilibili.com/video/BV1FjyHYmEDQ) |
| 视频/博客/代码 | [学习资料全景索引](./resources/links/资源索引.md) |
| 全部 PDF | [论文索引](../paper/论文索引.md) |
| 算法层补充 | [VLA 算法层清单](./VLA算法层学习路线与论文清单.md) |

---

## 十四、给小白的一句忠告

> **不要跳过 Phase 3 直接训大模型。**  
> 业界 2026 的共识是：瓶颈在 **数据协议与清洗**，不在「再堆 1B 参数」。  
> 先把 **两种不同格式的数据映射进同一 Unified Action、能稳定混合训练**，你们就已经走在 RDT 验证过的正确路上了。

---

*本路线随仓库论文/笔记更新；发现新的优质视频或博客请补进 [资源索引](./resources/links/资源索引.md)。*
