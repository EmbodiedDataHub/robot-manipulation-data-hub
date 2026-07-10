---
title: "数采到 VLA 训练 · 数据管线整体方案"
tags:
  - data-pipeline
  - data-collection
  - VLA
  - RLDS
  - LeRobot
  - unified-format
  - data-engine
updated: "2026-07-10"
related_notes:
  - "./机器人数据工作综合调研报告.md"
  - "./VLA训练与数据全貌-深度版.md"
  - "./paper-note/VLA-Datasets-Benchmarks-Data-Engines.md"
  - "./paper-note/IL-Paradigms/概述.md"
  - "./机器人操作数据学习路线报告.md"
reference_outline: "https://share.mubu.com/doc/37epqXwHRFD"
---

# 数采到 VLA 训练 · 数据管线整体方案

> **目标**：调研当前主流方案，给出 **从数据采集 → 清洗对齐 → 格式统一 → VLA 训练** 的可落地整体架构。  
> **范围**：基于本仓库 **49+ 篇论文**、9 篇综述及 2024–2026 工程实践；对齐用户参考大纲 [幕布 · 数采到训练](https://share.mubu.com/doc/37epqXwHRFD)。  
> **读者**：数据采集工程师、数据平台、VLA 训练工程师。

---

## 一、Executive Summary

### 1.1 核心结论

| 问题 | 2026 共识答案 |
|------|--------------|
| 瓶颈在哪？ | 不在「有没有 demo」，而在 **质量、对齐、多样性、失败覆盖、语言标注** |
| 格式怎么选？ | **采集期 LeRobot / MCAP** → **贡献 OXE 用 RLDS** → **训练期按模型 loader 导出** |
| 动作怎么统一？ | 没有全球唯一标准；工程上采用 **Canonical Schema + 128-d Unified Action（RDT）或 7-d EEF delta（OpenVLA）** 两层 |
| 数采路线？ | **五条互补范式**（Teleop / 手持 / VR / Ego / 仿真），用 **Data Pyramid** 分层混合 |
| 统一的关键？ | **中间表示（IR）** 比「直接存成某一种训练格式」更重要 |

### 1.2 推荐整体架构（一句话）

```text
多源采集 → Canonical Episode IR → QA/标注/Action 变换 → 双轨导出(RLDS+LeRobot) → 模型专用 Collator → VLA 训练
```

**不要**让 ALOHA 存 HDF5、UMI 存 zarr、VR 存 ROS bag 后直接混训——先收敛到 **Canonical IR**。

---

## 二、端到端管线总览

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 6 · VLA 训练                                                          │
│  OpenVLA(离散) · π0(Flow) · RDT(DiT) · ACT · Diffusion Policy               │
│  消费: batch = {images, language, action[, proprio, embodiment_id]}         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 5 · 训练混合 & Recipe                                                 │
│  Data Pyramid · mixing weight · 负迁移诊断 · unified action 映射              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 4 · 格式导出（双轨）                                                   │
│  LeRobot (Parquet+MP4, HF 微调)  │  RLDS/TFRecord (OXE 预训练贡献)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3 · Canonical Episode IR（本方案核心）                                 │
│  统一 step schema · 时间戳 · 相机命名 · action 物理语义 · success · language  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 2 · 清洗 / 对齐 / 变换                                                 │
│  同步(<10ms) · QA 过滤 · latency shift · FK/IK · delta · 归一化 · 分段标注    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1 · 多源采集（五条范式）                                               │
│  A Teleop  B 手持  C VR  D Ego  E 仿真/Data Engine                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↑
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 0 · 协议设计（采集前必须定稿）                                          │
│  任务 · 传感器 · 控制频率 · action 定义 · 成功标准 · 多样性 protocol         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 训练接口的统一性

不管数据从哪来，VLA/IL 训练的 **逻辑样本** 始终是：

```text
样本 = (o_t, ℓ, a_t)  或 chunk  (o_t, ℓ, a_{t:t+H})

o_t : 视觉观测（+ 可选 proprio）
ℓ   : 语言指令（episode 或 step 级）
a_t : 机器人动作（连续向量 或 离散 token）
```

**差异只在 `(o_t, a_t)` 如何产生、如何变换、如何 pad 到模型维度。**

---

## 三、五条数据采集范式（调研 + 本仓库论文）

### 3.1 范式总表

| 范式 | 代表（本仓库） | 需要机器人 | Fidelity | Scale | 原始信号 | 典型 Action |
|:----:|----------------|:----------:|:--------:|:-----:|----------|-------------|
| **A. 实验室 Teleop** | ALOHA, GELLO, Mobile ALOHA, DROID | ✅ | ⭐⭐⭐⭐⭐ | 小 | leader 关节 / 从端 executed cmd | joint / EEF |
| **B. 手持野外** | UMI, FastUMI, DexUMI, DexWild | ❌ | ⭐⭐⭐⭐ | 中→大 | GoPro RGB + SLAM pose + gripper | relative EEF delta |
| **C. VR / 人形 Teleop** | Open-TeleVision, OPEN TEACH, TWIST2 | ✅ | ⭐⭐⭐⭐ | 中 | VR 6D pose → IK → joint | whole-body joint |
| **D. Ego / 互联网视频** | EgoVLA, EgoScale, Phantom, EgoBridge | ❌ | ⭐⭐ | 极大 | RGB 视频，无 robot action | 需 retarget / latent |
| **E. 仿真 / 合成** | MimicGen, RoboGen, LIBERO | ❌(虚拟) | ⭐⭐⭐ | 极大 | sim state + render | 与 sim 一致 EEF/joint |

**本地论文索引**：[`paper/论文索引.md`](../paper/论文索引.md) §1–§5 Data Acquisition

---

### 3.2 范式 A · 实验室 Teleop

**流程**：

```text
人操作主端 ──映射──→ 机器人从端执行
     │                    │
     └──── 同步录制 ────────┘
           每步: {cameras[], proprio, executed_action, timestamp}
```

| 子类型 | 映射方式 | 本仓库论文 | Action 特点 |
|--------|----------|-----------|-------------|
| **运动学同构** | 主从结构一致，关节一一对应 | ALOHA, ALOHA 2 | 14-d joint + gripper |
| **外骨骼读角** | 被动关节 → 线性映射 | GELLO | 通用 arm，低成本 |
| **移动 whole-body** | 底座 + 双臂 | Mobile ALOHA, TidyBot++ | 高维 joint |
| **桌面 fleet** | VR/3D mouse → IK | DROID, RT-1 风格 | 7-d EEF delta/abs |

**前处理要点**：

1. 录 **executed action**（机器人实际收到的 command），不是 leader raw（若有滤波/限速）
2. **时间对齐**：多相机 + proprio + action，偏差 **< 10ms**
3. **Latency compensation**：teleop 延迟 50–200ms，高级管线做 timestamp shift 或 action 对齐到「决策时刻」的 obs
4. **控制频率**：常见 20–50 Hz；与 VLA chunk 设计一致

**笔记**：[ALOHA 概述](./paper-note/ALOHA/概述.md) · [GELLO/DROID 见论文索引 §1–§2](../paper/论文索引.md)

---

### 3.3 范式 B · 手持野外（UMI 系）

**核心**：采集时 **没有机器人**，只有人手持夹爪 + 相机。

```text
GoPro RGB + 夹爪开合 + SLAM 位姿
        ↓ 离线
推断 EEF 6D 轨迹 + gripper → 合成 (o_t, a_t)
        ↓ 部署
同构夹爪 + 相机装到真机 → 策略迁移
```

| 论文 | 改进点 |
|------|--------|
| **UMI** | 鱼眼+侧镜、SLAM、relative delta、latency matching |
| **FastUMI** | T265 替代 VIO，减遮挡失败 |
| **DexUMI** | 灵巧手扩展 |
| **DexWild** | 纯人手，更极端 scale |
| **RDT2 UMI** | CNC 夹爪 + 红外追踪，10k 小时 fleet |

**前处理要点**（UMI 特有）：

```text
1. SLAM / 追踪 → 相机 6D pose 序列
2. 手眼标定 → camera pose → EEF pose（刚性偏移）
3. gripper 状态检测（视觉分类或编码器）
4. 转 relative delta action（抗 SLAM 漂移）
5. latency matching：训练时模拟部署延迟
6. QA：去掉 tracking lost 的 episode
```

**笔记**：[UMI](./paper-note/UMI-Universal-Manipulation-Interface.md) · [RDT2 §9](./paper-note/RDT-Foundation-Models.md)

---

### 3.4 范式 C · VR / 人形 Teleop

**流程**：

```text
VR 头显 + 手柄/手套 → 人体 6D pose
        ↓ IK + retarget
机器人 joint targets（whole-body）
        ↓
同步录制 主动立体头显 RGB + joint state
```

| 论文 | 特点 |
|------|------|
| **Open-TeleVision** | **主动立体**头显相机，60Hz，H1/GR-1 |
| **OPEN TEACH** | 开源 VR 框架，~$500 |
| **TWIST2** | 无 mocap 人形 whole-body 采集 |

**与 ALOHA 差异**：

| | ALOHA Teleop | Open-TeleVision |
|---|-------------|-----------------|
| 映射 | 关节同构 | VR pose → **IK** |
| 相机 | 固定腕部/外部 | **头戴主动立体** |
| 机器人 | 双臂 ViperX | 人形 whole-body |
| 策略 | ACT joint chunk | ACT variant (DinoV2+stereo) |

**前处理要点**：

1. IK 失败 / 奇异点 episode 需 QA 过滤
2. 立体双目存 **left/right** 或 stacked，与训练一致
3. whole-body 高维 action → Unified Action Space 或 embodiment-specific head

**本地 PDF**：`paper/Data Acquisition/VR Teleoperation/`

---

### 3.5 范式 D · Ego / 互联网视频

**原始数据**：RGB 视频 +（可选）字幕；**无 robot action、无精确 pose**。

**四条「翻译」路径**（[VLA 数据综述 §3.1](./paper-note/VLA-Datasets-Benchmarks-Data-Engines.md)）：

| 路径 | 代表 | 输出 |
|------|------|------|
| 结构重建 | Video2Policy | GPT 语义 → 仿真任务代码 → (o,a) |
| 视觉 inpainting | H2R | 人手→机械手，提取视觉运动 |
| 物理 retarget | RoboWheel, EgoBridge | 手 pose → SDF 优化 → 可行轨迹 |
| Latent / affordance | Phantom, EgoVLA | 不直接输出 motor，学 latent action |

**2026 共识**：

> Ego 数据 = **Tier 1 预训练 prior**；Finetune **必须**补 robot-aligned teleop/UMI（通常 hundreds 小时级）。

**本仓库 Ego 论文**（13 篇）：EgoMimic, EgoVLA, Phantom, EgoBridge, EgoScale, EgoZero 等 → [`论文索引 §5`](../paper/论文索引.md)

---

### 3.6 范式 E · 仿真 / Data Engine

**教师类型**：

| 教师 | 代表 | 数据特点 |
|------|------|----------|
| 脚本/规划 | RLBench, robosuite | 规则生成，量大 |
| RL 专家 | RoboGen | 任务级 RL 后导出 |
| Demo 扩增 | **MimicGen** | 200 种子 → 50K 变体 |
| LLM 生成任务 | RoboGen, GenSim | 零人工任务设计 |

**前处理要点**：

- sim action 空间与 deploy **必须一致**（通常 EEF delta）
- domain randomization（光照、摩擦、纹理）
- 自动 success 标签（sim 优势）
- VLM critic 过滤不合理轨迹（2026 实践）

**本地 PDF**：`paper/Data Acquisition/Data Generation/MimicGen...`

---

### 3.7 范式选型决策树

```text
需要精确力控 / 接触-rich？
├─ 是 → A (teleop+力传感) 或 B+力传感 co-design
└─ 否 → 需要 cross-embodiment 野外泛化？
         ├─ 是 → B (UMI) 或 D→B 混合（Ego 预训练 + UMI 微调）
         └─ 否 → 有 lab 机器人预算？
                  ├─ 是 → A（双臂 ALOHA / 单臂 DROID）
                  ├─ 人形 whole-body → C (VR)
                  └─ 否 → E (仿真 bootstrap) + 少量 B 校准
```

---

## 四、格式标准调研：RLDS / LeRobot / HDF5 / MCAP

### 4.1 格式对比

| 格式 | 定位 | 优点 | 缺点 | 何时用 |
|------|------|------|------|--------|
| **RLDS** | Google OXE 标准 | 统一 episode/step schema、TFDS、跨 lab | TF 生态、转换成本高 | **OXE 预训练、跨机构贡献** |
| **LeRobot** | HuggingFace 机器人 ML | Parquet+MP4、Hub、PyTorch 友好 | 较新、老代码需 adapter | **微调、开源栈默认** |
| **HDF5** | ACT/DP 传统 | 随机访问、成熟 | schema 不统一 | **复现 ALOHA/ACT** |
| **Zarr** | Diffusion Policy | 大数组、chunk 友好 | 非通用交换格式 | **DP Push-T 等** |
| **MCAP/ROS2 bag** | 实时录制 | 多 topic、时间戳原生 | 需后处理 | **采集原始态** |
| **Robo-DM/EBML** | 超大规模 | 70× 压缩、50× 解码加速 | 生态较新 | **万小时级归档** |

**实践建议**（[综合调研报告 §4.1](./机器人数据工作综合调研报告.md)）：

```text
采集: MCAP/ROS2 或 LeRobot recorder（原生）
  ↓
Canonical IR（JSON/Parquet 逻辑 schema）
  ↓
导出双轨: LeRobot（HF 微调）+ RLDS（若贡献 OXE）
```

### 4.2 RLDS / Open X-Embodiment 核心结构

**论文**：Open X-Embodiment (2310.08864) · 本地 `paper/Surveys/`

```text
Dataset (TFDS)
 └── Episode
      ├── episode_metadata: {robot_type, file_path, ...}
      └── Steps (固定 length 或 ragged)
           ├── observation
           │    ├── image_0, image_1, ...  (uint8 HWC)
           │    ├── state                  (proprio, float32)
           │    └── natural_language_instruction (optional)
           ├── action                      (float32, dataset-specific dim)
           └── is_terminal / is_first
```

**OXE 规模**：1M+ episodes · 22 robots · 60 子数据集 · **异构 action dim 共存**

**关键经验**：

- RT-1-X 跨 robot 迁移 **+50%**；但 2026 后续工作警告 **盲目混合可能负迁移**
- 贡献 OXE 需实现 **TFDS DatasetBuilder**
- 常用归一化：action → **7-d EEF delta** + min/max 或 quantile norm

### 4.3 LeRobot 核心结构

**生态**：[`hugging-face.cn/docs/lerobot`](https://hugging-face.cn/docs/lerobot/il_robots)

```text
dataset/
 ├── meta/info.json          # fps, features, robot_type
 ├── meta/episodes.jsonl     # episode 索引
 ├── data/chunk-*/           # Parquet: state, action, timestamp
 └── videos/chunk-*/         # MP4 per camera key
```

**特点**：

- PyTorch `LeRobotDataset` 直接 `lerobot-train`
- Hub 一键分享
- 支持 ACT、Diffusion Policy、OpenVLA 等 recipe

### 4.4 本仓库各算法期望的输入

| 训练目标 | 典型格式 | Action 维 | 相机 |
|----------|---------|----------|------|
| ACT/ALOHA | HDF5 / LeRobot | 14-d joint | 4 路固定 |
| Diffusion Policy | Zarr / HDF5 | EEF delta, horizon H | 1–2 路 |
| OpenVLA | RLDS / LeRobot | 7×256 bins | 1–2 路 224² |
| π0 | 私有 / LeRobot 适配 | continuous chunk H=50 | 多路 |
| RDT-1B | RLDS + 自采 | **128-d unified** | 3 路 + T5 lang |

---

## 五、统一方案核心：Canonical Episode IR

### 5.1 为什么需要中间表示（IR）

| 问题 | 无 IR | 有 IR |
|------|-------|-------|
| ALOHA joint + UMI EEF + VR whole-body | 三套脚本，无法混训 | 先映射到 IR，再导出 |
| 改 action 定义（abs→delta） | 重录数据 | IR 层重算 action |
| 换训练框架（ACT→OpenVLA） | 重写 converter | IR → 多目标 exporter |
| QA / 标注 | 分散在各格式 | IR 层统一过滤 |

### 5.2 建议的 Canonical Schema（逻辑层）

```yaml
# episode level
episode_id: str
source: enum[teleop_aloha, teleop_vr, umi, ego, sim_mimicgen, ...]
robot_type: str                    # "aloha_viperx", "franka", "umi_gripper", ...
embodiment_id: int                 # 训练用 embedding
success: bool
language: str                      # 主指令
language_variants: list[str]       # 反事实重标注
metadata:
  lab: str
  scene_id: str
  objects: list[str]
  control_frequency_hz: float
  collection_date: str

# step level (t = 0..T-1)
steps:
  - timestamp_ns: int64            # 统一单调时钟
    observations:
      images:
        exterior_1: {uri or bytes ref, H, W, encoding}
        wrist_left: {...}
        wrist_right: {...}
      proprio:
        joint_pos: float32[Dj]      # optional
        eef_pose: float32[7]        # xyz + quat or 6D rot + gripper
        base_vel: float32[3]        # optional mobile
    action:
      # 物理语义分开存，导出时再组合
      joint_pos_target: float32[Dj]     # optional
      eef_delta: float32[6]             # optional
      eef_absolute: float32[7]          # optional
      gripper: float32[1]
      unified_128: float32[128]         # optional, RDT 预计算
    flags:
      is_teleop_idle: bool
      tracking_quality: float         # UMI SLAM 质量
```

**设计原则**：

1. **观测与动作物理语义分离存储**——不强制采集时就定一种 action
2. **timestamp_ns 全局单调**——一切对齐在 IR 层完成
3. **camera key 命名统一**——`exterior_*`, `wrist_*`, `head_stereo_*`
4. **source 字段可追溯**——provenance / Datasheet

### 5.3 两层 Action 统一策略

```text
Layer 1 · 物理语义层（IR 内）
  joint_pos / eef_delta / eef_absolute / gripper 分开字段
  用 FK/IK 在 IR 层互转，保留原始 executed 值

Layer 2 · 训练导出层（按模型）
  OpenVLA  → 7-d EEF delta, quantile norm, 256 bins
  ACT      → 14-d joint chunk
  RDT      → 128-d unified pad（见 RDT 笔记 §4）
  π0       → continuous chunk, dataset mean/std
```

**RDT 128-d Unified Action**（[RDT 笔记 §4](./paper-note/RDT-Foundation-Models.md)）：

| 槽位 | 物理量 |
|------|--------|
| [0,10) | 右臂 joint |
| [10,15) | 右 gripper |
| [30,39) | 右 EEF 6D |
| [50,60) | 左臂 joint（对称） |
| … | 速度、基座等 |

单臂/UMI/ALOHA 按规则 **填槽 + 补零**。

**OpenVLA / OXE 惯例**：7-d = Δxyz(3) + Δrpy(3) + gripper(1)

---

## 六、各源 → IR → 训练格式：转换 Recipe

### 6.1 ALOHA Teleop → IR → ACT

```text
Raw (ROS/HDF5):
  qpos (14), images (4), optional effort

→ IR:
  proprio.joint_pos = qpos
  action.joint_pos_target = qpos_{t+1} 或 executed cmd
  images: exterior + wrist_l + wrist_r

→ Export ACT:
  HDF5 episode / LeRobot
  chunk H=100, L1+KL 训练
```

### 6.2 UMI → IR → Diffusion Policy / RDT2

```text
Raw:
  GoPro MP4 + SLAM trajectory + gripper log

→ IR:
  SLAM QA filter
  eef_absolute from hand-eye calib
  action.eef_delta = relative(eef_t, eef_{t-1})
  latency: shift action 或 obs 模拟部署延迟

→ Export DP:
  zarr / LeRobot, horizon H, DDPM cond

→ Export RDT2:
  map eef+gripper to unified slots / UMI 原生 6D
```

### 6.3 Open-TeleVision VR → IR → ACT

```text
Raw:
  stereo head RGB (L/R) + whole-body joint targets

→ IR:
  images.head_stereo_left/right
  proprio + action.joint_pos_target (after IK)
  filter IK failure episodes

→ Export:
  ACT variant, 28D (H1) / 19D (GR-1) chunk
```

### 6.4 DROID / RT-1 风格 → IR → OpenVLA

```text
Raw:
  exterior + wrist, 7-d executed EEF

→ IR:
  action.eef_delta or absolute（与原始一致）
  language from operator or VLM relabel

→ Export RLDS:
  OXE DatasetBuilder schema
  action normalize to [-1,1] per dataset quantile

→ Export OpenVLA:
  7 × 256 bin tokenization
```

### 6.5 MimicGen Sim → IR → 混合预训练

```text
Raw:
  robosuite / MuJoCo rollout

→ IR:
  source=sim_mimicgen
  success auto-labeled
  domain_rand params in metadata

→ Export:
  与真机 IR 相同 schema → 混合时需 **embodiment_id** + **mixing weight**
```

### 6.6 Ego 视频 → IR（弱监督）

```text
Raw:
  MP4 only

→ IR (partial):
  仅 observation 有效；action 经 retarget 管线填入
  或 action = latent/placeholder，标记 supervision=weak

→ 用途:
  Tier 1 预训练（EgoVLA 类），不直接进入 OpenVLA BC 除非有 retarget 质量 QA
```

---

## 七、清洗与 QA 管线（八步标准）

综合 [VLA 训练全貌 §6](./VLA训练与数据全貌-深度版.md) + [综合调研报告](./机器人数据工作综合调研报告.md)：

| Step | 名称 | 输入→输出 | 关键检查 |
|:----:|------|-----------|----------|
| 1 | **Record** | Protocol → Raw bag | 50Hz、曝光、episode 边界 |
| 2 | **Synchronize** | Raw → aligned steps | drift **< 10ms**，丢帧 **< 1%** |
| 3 | **QA Filter** | episodes → 合格子集 | 去 idle、blur、joint limit、SLAM lost |
| 4 | **Segment & Annotate** | → +language, success | 避免模板化语言 |
| 5 | **Action Transform** | raw cmd → 目标表示 | FK/IK、delta、clip、norm |
| 6 | **Export** | IR → RLDS + LeRobot | 版本 tag |
| 7 | **Split** | 按 **场景/物体/房间** | ❌ 禁止随机 episode split |
| 8 | **Loader Verify** | 1 epoch + 可视化 overlay | action 与图像语义一致 |

**工程数字**：

- 无 QA 时 **30–40%** episode 不可用
- **去掉最差 20%** 往往优于 **多加 50% 数据**
- 操作员成本 **$30–50/hr**；1000 ep ≈ **$5k–15k**

---

## 八、语言标注与数据质量（常被忽视）

| 问题 | 现象 | 修复 |
|------|------|------|
| **Language impoverished** | 全是 `pick up the object` | GPT-4V 多模板 + 人工 spot check |
| **Counterfactual relabel** | 一条轨迹多种语言目标 | CAST 等方法 **+27pp** |
| **失败轨迹丢弃** | 只训 success | 保留 slip/collision，标 `success=false` |
| **模态缺失** | 无触觉/力 | RH20T、Koala 等；或标注 `force=null` |

**Qwen-RobotManip / ABot-M0 工程启示**（本仓库 FM 论文）：

- **Alignment-first**：Camera-centric delta、时间对齐优先于堆数据量
- **UniACT 清洗**：600 万轨迹异构 raw → 统一表示（#48 ABot-M0）

---

## 九、训练混合：Data Pyramid 与负迁移

### 9.1 Data Pyramid（RDT2 / 行业共识）

```text
        Tier 3 · 精调
        50–500 ep 目标机器人 teleop（决定最终成功率）
              ↑
        Tier 2 · 预训练
        OXE + UMI + MimicGen（决定泛化）
              ↑
        Tier 1 · 先验
        Ego 视频 + Web VLM co-train（决定语义 breadth）
```

### 9.2 混合 Recipe 示例

| 目标模型 | Tier 1 | Tier 2 | Tier 3 |
|----------|--------|--------|--------|
| OpenVLA finetune | OXE 子集 optional | — | 目标 robot 100–1K ep |
| RDT-1B | — | OXE 1M pretrain | ALOHA 6K ft |
| RDT2 | VLM 图文对 | UMI 10K hr | task-specific optional |
| π0 | Web co-train | 私有 10K hr mixed | 50–200 ep / task |

### 9.3 负迁移防护

- 采样权重 ∝ **√N** 或 **dynamic loss-based**（RDT-1B）
- **embodiment_id** embedding
- 诊断：单源 vs 混合 ablation
- **5% curated coreset** 可恢复 85–90% 全量性能（2026 文献）

---

## 十、落地架构：推荐团队分工

```text
┌─────────────────────────────────────────────────────────┐
│ 采集 App 层                                              │
│  ALOHA recorder │ UMI app │ VR teleop node │ Sim exporter│
└───────────────────────────┬─────────────────────────────┘
                            ↓ MCAP / native
┌─────────────────────────────────────────────────────────┐
│ 数据平台（核心投资）                                      │
│  ingest → sync → QA → IR store → annotate → export      │
│  技术栈: Python + Parquet/S3 + dagster/airflow            │
└───────────────────────────┬─────────────────────────────┘
                            ↓ LeRobot + RLDS
┌─────────────────────────────────────────────────────────┐
│ 训练层                                                   │
│  collator per model (OpenVLA / π0 / RDT / ACT)          │
└─────────────────────────────────────────────────────────┘
```

### 10.1 分阶段落地路线（12 周）

| 阶段 | 周 | 交付物 |
|------|:--:|--------|
| **P0 协议** | 1–2 | Protocol 文档：action 定义、相机、频率、命名 |
| **P1 单源闭环** | 3–5 | 选 ALOHA **或** UMI 一条链路 → IR → LeRobot → ACT/DP 训练 |
| **P2 双轨导出** | 6–7 | IR → RLDS builder；Hub 上传 |
| **P3 多源 ingest** | 8–9 | 接入第 2 源（sim 或 VR）；统一 QA |
| **P4 混合训练** | 10–11 | embodiment_id + mixing；OpenVLA 或 RDT 小规模实验 |
| **P5 文档化** | 12 | Datasheet + eval protocol 对齐部署场景 |

---

## 十一、论文阅读顺序（数据层专用）

| Phase | 时间 | 论文 | 本地 | 笔记 |
|:-----:|------|------|:----:|------|
| 0 | 2d | VLA Datasets/Benchmarks/Data Engines | ✅ | [数据综述](./paper-note/VLA-Datasets-Benchmarks-Data-Engines.md) |
| 0 | 1d | Open X-Embodiment | ✅ | 本文 §4.2 |
| 1 | 1w | ALOHA → GELLO | ✅ | [ALOHA](./paper-note/ALOHA/概述.md) |
| 2 | 1w | UMI → FastUMI | ✅ | [UMI](./paper-note/UMI-Universal-Manipulation-Interface.md) |
| 3 | 1w | Open-TeleVision → TWIST2 | ✅ | [Data-Pipeline/VR](./paper-note/Data-Pipeline/VR与人形Teleop数采.md) |
| 4 | 1w | MimicGen | ✅ | [Data-Pipeline/Sim](./paper-note/Data-Pipeline/仿真与合成数据引擎.md) |
| 5 | 2w | EgoVLA → EgoScale → EgoBridge | ✅ | [Data-Pipeline/Ego](./paper-note/Data-Pipeline/Ego视频数采与对齐.md) |
| 6 | 1w | RDT-1B unified action → RDT2 UMI scale | ✅ | [RDT](./paper-note/RDT-Foundation-Models.md) |
| 7 | 1w | Qwen-RobotManip / ABot-M0 | ✅ | 工程向，论文索引 §4 |
| 8 | 选读 | Robo-DM, DROID, BridgeData V2 | ✅ | [综合调研报告](./机器人数据工作综合调研报告.md) |

**算法层并行**： [IL 范式概览](./paper-note/IL-Paradigms/概述.md)

---

## 十二、本仓库论文 × 数采范式映射

| 范式 | 论文 # | 目录 |
|------|--------|------|
| Teleop | 26–32 | `paper/Data Acquisition/Robot Teleoperation/` |
| 手持 UMI | 33–40 | `paper/Data Acquisition/Hand-Held Grippers Teleoperation/` |
| VR 人形 | 41–43 | `paper/Data Acquisition/VR Teleoperation/` |
| Ego 人类 | 44, 50–60 | `paper/Data Acquisition/Ego Human Data/` |
| 仿真生成 | 4, 等 | `paper/Data Acquisition/Data Generation/` |
| 数据格式/OXE | 2, 综述 | `paper/Surveys/` |
| FM 数据工程 | 47–48 | `paper/Foundation Models/` |

完整可点击索引：[论文索引.md](../paper/论文索引.md)

---

## 十三、开放问题与建议

| # | 问题 | 建议 |
|---|------|------|
| 1 | Unified Action 无全球标准 | IR 存物理语义 + 多 exporter；预训练选 RDT 128-d **或** OXE 7-d 之一为主 |
| 2 | Ego 最小 robot 校准量 | 假设 hundreds hr teleop/UMI 做 Tier 3，Ego 仅 Tier 1 |
| 3 | 格式战争 RLDS vs LeRobot | **双轨导出**，IR 层统一 |
| 4 | 负迁移 | 混合前必做 ablation + 采样权重调参 |
| 5 | 评估不可信 | 数据改进需配 **扰动 eval**（LIBERO-Plus 类） |

---

## 十四、关联文档索引

| 文档 | 用途 |
|------|------|
| [本文](./数采到VLA训练-数据管线整体方案.md) | **总方案** |
| [Data-Pipeline 概述](./paper-note/Data-Pipeline/概述.md) | 数采子主题索引 |
| [机器人数据工作综合调研报告](./机器人数据工作综合调研报告.md) | 2026 行业调研 |
| [VLA 训练与数据全貌（深度版）](./VLA训练与数据全貌-深度版.md) | 训练 recipe + batch 形态 |
| [机器人操作数据学习路线报告](./机器人操作数据学习路线报告.md) | ALOHA→UMI→OTV→RDT 主线 |
| [IL 范式概览](./paper-note/IL-Paradigms/概述.md) | 算法层阅读顺序 |
| [资源索引](./resources/links/资源索引.md) | 视频/博客/LeRobot 教程 |

---

> **下一步行动**：先定 Protocol（Layer 0）→ 实现 **单源 IR 闭环**（ALOHA 或 UMI 二选一）→ 再扩展多源 ingest 与混合训练。
