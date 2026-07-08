# 现代 VLA 训练与数据全貌（深度版）

> **目标**：相对深入地建立全局认知——现代 VLA **怎么一步步训出来**、**需要什么数据**、**数据怎么产生和处理**。  
> **读者**：已有 Day 0 基础、想从「知道名词」到「能画出完整流水线」的同学。  
> **关联**：[算法层论文清单](./VLA算法层学习路线与论文清单.md) · [数据调研报告](./机器人数据工作综合调研报告.md) · [论文索引](../paper/论文索引.md)

---

## 目录

1. [一张总流程图：从 0 到部署](#一一张总流程图从-0-到部署)
2. [现代 VLA 训练六阶段（逐步拆解）](#二现代-vla-训练六阶段逐步拆解)
3. [VLA 模型内部：四个模块 + 三种 Action Head](#三vla-模型内部四个模块--三种-action-head)
4. [数据长什么样：一条训练样本的 anatomy](#四数据长什么样一条训练样本的-anatomy)
5. [数据从哪来：五条产线 × 处理细节](#五数据从哪来五条产线--处理细节)
6. [数据处理流水线：Raw → Trainable（八步）](#六数据处理流水线raw--trainable八步)
7. [三种典型训练 Recipe（对照表）](#七三种典型训练-recipe对照表)
8. [部署闭环：推理频率 vs 控制频率](#八部署闭环推理频率-vs-控制频率)
9. [2026 共识：Data Pyramid 与负迁移](#九2026-共识data-pyramid-与负迁移)
10. [和本地仓库的衔接](#十和本地仓库的衔接)

---

## 一、一张总流程图：从 0 到部署

把「训出一个能用的 VLA」想成 **开餐厅**：

| 环节 | 餐厅类比 | VLA 对应 |
|------|---------|----------|
| 菜谱设计 | 定做什么菜、用什么料 | **Protocol Design**：任务、传感器、action 表示 |
| 采购 | 去不同供应商进货 | **多源采集**：Teleop / UMI / Ego / Sim |
| 洗菜切菜 | 去坏叶、统一规格 | **QA + 归一化 + 语言标注** |
| 中央厨房 | 大批量预制 | **Pretrain**：OXE / 混合万小时数据 |
| 门店微调 | 按本地口味调整 | **Finetune**：目标机器人 + 目标任务 |
| 出餐 | 顾客点菜 → 上菜 | **Deploy**：语言指令 → 50Hz 控制 |

```
                         ┌──────────────────────────────────────┐
                         │  Stage 0: VLM 底座（通常已有）          │
                         │  SigLIP / PaliGemma / Qwen-VL        │
                         └──────────────────┬───────────────────┘
                                            │ 加载预训练权重
┌─────────────── 数据侧 ────────────────────┼─────────── 模型侧 ───────────────┐
│                                           ▼                                   │
│  ① Protocol  →  ② 采集  →  ③ QA/标注  →  ④ 格式转换  →  ⑤ 混合 Recipe        │
│     │              │            │              │              │               │
│     │         Teleop/UMI     过滤20-40%      RLDS/LeRobot    Data Pyramid      │
│     │         Ego/Sim      语言重标注      归一化/切块                      │
│     └──────────────────────────────────────────────────────────┘               │
│                                           │                                   │
│                                           ▼                                   │
│                              ⑥ Pretrain（跨机器人通才）                        │
│                                           │                                   │
│                                           ▼                                   │
│                              ⑦ Finetune（目标 embodiment + 任务）              │
│                                           │                                   │
└───────────────────────────────────────────┼───────────────────────────────────┘
                                            ▼
                              ⑧ Deploy：Chunk 推理 + 低层控制器
```

**核心结论（先记住这三句）：**

1. **VLA ≠ 从零训练大模型**：绝大多数工作是 **VLM 底座 + Action Head + 机器人数据**，底座往往冻结或 LoRA。
2. **数据流水线与模型训练同等重要**：同样 7B 参数，QA 差的数据 vs 精洗数据，成功率可差 30pp+。
3. **Pretrain 和 Finetune 用的数据形态不同**：Pretrain 要 **广**（多机器人、多任务）；Finetune 要 **深**（目标场景高质量 demo）。

---

## 二、现代 VLA 训练六阶段（逐步拆解）

### Stage 0 · VLM 预训练（通常不自己做）

**做什么**：在互联网图文对上训练「看懂图 + 理解语言」的能力。

**代表底座**：SigLIP（OpenVLA）、PaliGemma（π0）、PaLI-X（RT-2）、Qwen-VL（Qwen-RobotManip）

**机器人团队的实际动作**：
- 直接下载 HuggingFace 上的预训练权重
- 通常 **冻结 Vision Encoder**，只训融合层 + Action Head（省算力、防 catastrophic forgetting）
- RT-2 例外：做 **co-fine-tune**，机器人数据 + 部分 web VLM 数据一起更新

**你要理解的程度**：知道 VLM 提供了「语义理解」和「视觉泛化」，VLA 在此基础上加「动手」。

---

### Stage 1 · 机器人数据 Pretrain（学「通才操作」）

**目标**：一个模型见过很多机器人、很多任务，学到 **通用 manipulation prior**。

**典型数据规模**：

| 项目 | 预训练数据量 | 来源 |
|------|-------------|------|
| RT-1 | ~130K episodes | 单机构 Everyday Robots 车队 |
| OpenVLA | ~970K episodes | Open X-Embodiment 子集 |
| Octo | ~800K episodes | OXE + 自建混合 |
| RDT-1B | ~1M+ episodes | 多源 + Unified Action Space |
| π0 | 未完全公开，估 **10K+ 小时** | 跨多种真实机器人 |

**训练目标（监督学习）**：

```
给定:  o_t = {图像_1..K, 语言指令, 可选: 关节状态}
预测:  a_{t:t+H} = 未来 H 步动作（Action Chunk）

Loss = MSE(预测动作, 专家动作)           # 连续头 (Diffusion/Flow/MLP)
  或 = CrossEntropy(预测 token, 动作token)  # 离散头 (RT-2/OpenVLA)
```

**关键技术选择**：

| 选择 | 选项 A | 选项 B | 谁用 |
|------|--------|--------|------|
| Action 表示 | 离散 token（256 bins） | 连续向量 + Diffusion/Flow | RT-2 / OpenVLA vs π0 / RDT |
| Chunk 长度 H | 25–50 步 | 100+ 步 | OpenVLA ~8 步；ACT ~100 步 |
| 图像帧 | 单帧 | 多帧 history（2–3 帧） | 多数 VLA 用 1–2 帧 |
| 语言 | 每 episode 一句 | 每步不同 sub-instruction | CALVIN 式长任务用 sub-goal |

**Pretrain 产出**：一个 **通用 checkpoint**，能 zero-shot 或 few-shot 迁移到新机器人，但成功率通常不够高（30–60%），必须 Finetune。

---

### Stage 2 · Co-training（可选，RT-2 路线）

**做什么**：把 **机器人轨迹数据** 和 **互联网 VLM 数据** 混在一个 batch 里训。

**为什么**：
- 防止模型在大量机器人数据上训完，**忘了** VLM 的语义能力（catastrophic forgetting）
- 让机器人能利用 web 知识（「 extinct animal」→ 认识渡渡鸟玩具）

**RT-2 做法**：
- Batch 内混合：~50% 机器人 (image, instruction, action tokens)，~50% web VLM (image, QA text)
- 动作离散化为与文本共享词表的 token

**2026 趋势**：OpenVLA / π0 更常用 **冻结 VLM + 只训 action 相关层**，co-training 变少，但 **语言增强标注**（VLM 自动生成 diverse instruction）变多。

---

### Stage 3 · Finetune（学「在这个机器人上做好」）

**数据**：目标平台 **50–500 条** 高质量 teleop demo 即可显著提升（ALOHA 经验）；工业场景常 **1K–10K episodes**。

**Finetune 策略**：

| 策略 | 说明 | 适用 |
|------|------|------|
| **Full finetune** | 全部参数更新 | 数据多（>5K ep）、算力足 |
| **LoRA / Adapter** | 只训低秩矩阵 | OpenVLA 默认推荐；数据少 |
| **Action Head only** | 冻结 VLM，只训最后一层 | 快速验证、新 gripper |
| **Task-specific head** | 共享 backbone，每任务一个头 | 多任务但数据不均衡 |

**数据要求比 Pretrain 更严**：
- 相机位置 **与部署一致**（finetune 用 wrist cam，deploy 也必须 wrist cam）
- 成功轨迹为主，但 **5–10% 失败轨迹** 可提升鲁棒性（2026 新方向）
- 语言指令要 **多样化**（同一任务至少 10 种说法，防 language impoverishment）

---

### Stage 4 · Post-training（可选）

**RL / RLHF 式微调**：在仿真或真机用 RL 优化 success rate（仍属少数派，算力贵）。

**Test-time adaptation**：部署时在线收集失败 case 回灌（Data Engine 闭环）。

**Distillation**：大 VLA → 小 VLA（SmolVLA），用于边缘部署。

---

### Stage 5 · Deploy（部署闭环）

见 [第八节](#八部署闭环推理频率-vs-控制频率)。

---

## 三、VLA 模型内部：四个模块 + 三种 Action Head

### 3.1 四模块架构（以 OpenVLA 为例）

```
输入图像 ──→ [Vision Encoder: SigLIP ViT] ──→ 图像 token (256 个)
                                                    │
输入语言 ──→ [Tokenizer + Embedding] ──────────────┼──→ [Projector / Fusion]
                                                    │           │
可选关节状态 ──→ [Proprio MLP] ────────────────────┘           │
                                                                 ▼
                                                    [LLM Backbone: Llama 2, 7B]
                                                                 │
                                                                 ▼
                                                    [Action Head] → 动作
```

**各模块训练策略**：

| 模块 | Pretrain 时 | Finetune 时 |
|------|------------|------------|
| Vision Encoder | 通常 **冻结** | 通常冻结；数据域差大时可解冻最后几层 |
| LLM Backbone | 冻结或 LoRA | LoRA 为主 |
| Projector | **全训** | 全训 |
| Action Head | **全训** | **全训**（最关键） |

### 3.2 三种 Action Head（必须分清）

#### 类型 A · 离散 Autoregressive Token（RT-2, OpenVLA）

```
连续动作 [x, y, z, roll, pitch, yaw, gripper]
    ↓ 每维 uniform 分成 256 bins
离散 token 序列: [tok_142, tok_87, tok_201, ...]
    ↓ 与文本 token 共用 Transformer 词表
自回归生成，类似 GPT 写字
```

- **优点**：与 VLM 架构统一；天然支持 co-training
- **缺点**：量化误差；长 horizon 自回归慢
- **OpenVLA**：7 维动作 × 256 bins → 7 个 token / step

#### 类型 B · Diffusion / Flow Matching（π0, RDT, Diffusion Policy）

```
条件: 融合后的 visual-language embedding
过程: 从噪声 a_T ~ N(0,I) 逐步 denoise → a_0（真实动作）
训练: 预测 noise 或 velocity field（Flow Matching）
推理: 10–20 步 denoise，输出 H×dim 动作矩阵
```

- **优点**：连续动作、多模态（同一场景多种合理轨迹）；适合精细操作
- **缺点**：推理比 token 慢；需要更多调参
- **π0**：Flow Matching + 预测 **50 步 × 动作维** chunk

#### 类型 C · MLP / LSTM Direct Regression（ACT, 早期 BC）

```
融合特征 → MLP → 直接输出 a_{t:t+H}
```

- **优点**：简单、快
- **缺点**：多模态差（会取平均）；长 horizon 误差大
- **ACT 改进**：加 CVAE latent z 建模多模态 + Chunking

### 3.3 三代 VLA 对照（建立「谱系感」）

| 代际 | 代表 | 底座 | Action Head | 数据规模 | 核心贡献 |
|------|------|------|-------------|---------|---------|
| Gen-1 | RT-1 | 自建 EfficientNet+TokenLearner | Discrete token, Transformer | 130K ep | 多任务单模型 |
| Gen-2 | RT-2 | PaLI-X | 离散 token + co-train web | 130K + web | VLM 知识迁移 |
| Gen-2.5 | OpenVLA / Octo | 开源 VLM (SigLIP+Llama) | 离散 / Diffusion | ~1M (OXE) | 开源可复现 |
| Gen-3 | π0 / RDT2 | PaliGemma / DiT | Flow / Diffusion | 10K+ 小时 | 连续精细操作 +  scale |
| Gen-3+ | Qwen-RobotManip | Qwen-VL | 未完全公开 | 自建中文数据 | 中文指令 + 数据管线 |

---

## 四、数据长什么样：一条训练样本的 anatomy

### 4.1 Episode / Step 层级

```
Dataset
 └── Episode (一次任务尝试，如「把红杯子放到左边」)
      ├── metadata: {robot_type, lab, date, success, instruction_text, ...}
      └── Step × T  (通常 50–500 步，20–60 秒)
           ├── observation
           │    ├── image_exterior: (H, W, 3) uint8
           │    ├── image_wrist: (H, W, 3) uint8      # 可选
           │    ├── proprio: (D,) float                 # 关节角或 EEF pose
           │    └── language: str                       # 通常 episode 级
           └── action
                └── (D_action,) float                  # 下一步或 chunk 目标
```

### 4.2 不同项目的 D_action 维度

| 项目 | Action 空间 | 维度 | 说明 |
|------|------------|------|------|
| RT-1 / Bridge | Delta EEF + gripper | 7 | Δxyz(3) + Δrpy(3) + gripper(1) |
| ALOHA | Joint positions | 14 | 双臂各 6 关节 + 2 夹爪 |
| UMI | Relative EEF + gripper | 7 | 相对位移，与绝对定位解耦 |
| RDT | Unified bimanual | 128 | 统一编码后 pad/truncate |
| DROID | Absolute EEF | 7 | 绝对位姿 + gripper |

### 4.3 训练时一个 batch 的实际形态（OpenVLA 风格）

```python
# 伪代码 — 理解 dataloader 在干什么
batch = {
    "pixel_values": Tensor[B, 3, 224, 224],      # 图像，已 resize+normalize
    "input_ids": Tensor[B, seq_len],              # 语言 tokenized
    "labels": Tensor[B, 7],                       # 7 个 action token ids
    "actions_raw": Tensor[B, 7],                  # 原始连续动作（算 loss 用）
}
# 图像增强（仅训练）: 随机 crop, color jitter, 轻微 blur
# 动作归一化: 通常按数据集统计 mean/std 或 min/max 到 [-1,1]
```

### 4.4 语言标注从哪来

| 方式 | 做法 | 优缺点 |
|------|------|--------|
| **模板** | `"pick up the {object}"` | 快但 language impoverished |
| **人工** | 操作员口述或事后写 | 质量高但贵 |
| **VLM 自动生成** | GPT-4V 看首帧生成 10 种指令 | OpenVLA 推荐；需 spot check |
| **Counterfactual relabel** | 同一轨迹配不同语言目标 | CAST 等方法 +27pp |

---

## 五、数据从哪来：五条产线 × 处理细节

### 产线 A · 真机 Teleop（金标准）

**怎么产生**：
```
人 ──(leader arm / VR / 3D mouse)──→ 机器人跟做
         同步录制: 所有相机 + joint state + 时间戳
         每个 episode: 按空格/脚踏开始结束
```

**原始格式**：ROS2 bag / MCAP / HDF5 / LeRobot 原生

**处理要点**：
1. **时间对齐**：所有 sensor timestamp 偏差 < 10ms（用硬件 trigger 或 post-hoc 插值）
2. **Action 定义**：通常录 **executed action**（机器人实际执行的 command），不是 leader 的 raw input
3. **Latency compensation**：遥操作有 50–200ms 延迟，高级 pipeline 做 timestamp shift

**本地仓库对应**：ALOHA, GELLO, DROID 风格 → [论文索引 §1](../paper/论文索引.md#1-aloha)

---

### 产线 B · 手持野外（UMI 系）

**怎么产生**：
```
GoPro + 手持夹爪 → 人在真实环境操作 → 只录视频+夹爪开合
      ↓ 离线 SLAM
EEF 6D 轨迹 + gripper state → 合成 (observation, action) 对
```

**处理 pipeline（UMI 特有）**：
```
1. ORB-SLAM3 / 类似 → 相机 6D pose 序列
2. 手眼标定 → 相机 pose → EEF pose
3. 视频分类器 → gripper open/close
4. 转 relative delta action（抗 SLAM 漂移）
5. 部署时用同一夹爪 geometry 缩小 sim2real gap
```

**关键差异**：动作是 **推断的**，不是 **执行的** → fidelity 低于 Teleop，但 scale 和场景多样性更好。

**本地仓库对应**：[UMI 笔记](./paper-note/UMI-Universal-Manipulation-Interface.md)

---

### 产线 C · 仿真 / 合成（MimicGen 系）

**怎么产生**：
```
少量人类 demo (200) → MimicGen 在仿真里变换物体位姿/纹理
                    → 自动生成 50,000 条成功轨迹
```

**处理要点**：
- 仿真 action 空间与真机 deploy 空间 **必须一致**（通常 EEF delta）
- 需要 **domain randomization**（光照、摩擦、质量）防 overfit
- VLM critic 过滤物理不合理轨迹（2026 新实践）

**本地仓库对应**：MimicGen 论文已收录

---

### 产线 D · 人类 Ego 视频

**怎么产生**：
```
第一人称视频（Ego4D / 自采）→ 无 robot action label
```

**处理 pipeline（最难）**：
```
路径1 Hand pose 估计 → retarget 到机器人 → 物理优化 (RoboWheel)
路径2 视频 inpainting → 人手变机械手 → 提取视觉运动 (H2R)
路径3 VLM 语义解析 → 生成仿真任务代码 → sim rollout (Video2Policy)
路径4 只学 latent action / affordance，不直接输出 motor command (Phantom)
```

**2026 共识**：Ego 数据主要用于 **Pretrain prior**，Finetune 必须补 **robot-aligned** 数据。

**本地仓库对应**：[论文索引 §5](../paper/论文索引.md#5-ego)

---

### 产线 E · 跨机构混合（OXE / RDT）

**怎么产生**：各 lab 按 RLDS schema 贡献子数据集 → 中央聚合

**核心处理：Unified Action Space（RDT 做法）**：
```
Franka 7-DoF delta  ──┐
WidowX 7-DoF delta  ──┼──→ 统一 pad 到 D_max=128 维
ALOHA 14-DoF joint  ──┤     + embodiment id embedding
人形 whole-body     ──┘     + 归一化到 [-1,1]
```

**负迁移风险**：盲目混合所有数据可能 **降低** 单平台性能 → 需要 **Data Pyramid** 和 **mixing recipe** 调参。

---

## 六、数据处理流水线：Raw → Trainable（八步）

这是 **数据工程师** 的主战场，逐步对应：

```
Step 1 · 录制 (Record)
  输入: Protocol 文档
  输出: Raw bag/MCAP/HDF5
  检查: 采样率(20-50Hz)、分辨率、曝光、episode 边界清晰

Step 2 · 同步 (Synchronize)
  输入: Raw 多 topic 流
  输出: 对齐的 step 序列
  检查: max timestamp drift < 10ms；丢帧率 < 1%

Step 3 · 质检 (QA Filter)
  输入: 对齐 episode
  输出: 合格 episode 子集
  规则示例:
    - 去掉 idle 超过 3s 的段
    - 去掉 joint velocity 超 limit 的（操作失误）
    - 去掉 blur score 过高的帧
    - 去掉 SLAM tracking lost（UMI）
  经验: 去掉最差 20-40% 比多加 50% 数据更有效

Step 4 · 分段 & 标注 (Segment & Annotate)
  输入: 合格 episode
  输出: 带 language + success flag 的 episode
  注意: 避免全用 "pick up the object" 模板

Step 5 · Action 变换 (Transform)
  输入: raw joint commands
  输出: 目标 action 表示（EEF delta / unified space）
  操作: FK/IK、delta 计算、clip 到 workspace、归一化

Step 6 · 格式导出 (Export)
  输入: 标准化 episode
  输出: RLDS tfrecord / LeRobot parquet+mp4
  双轨: LeRobot 用于 HF 微调；RLDS 用于 OXE 贡献

Step 7 · 划分 (Split)
  按 **场景/物体/房间** 划分 train/val/test
  ❌ 不要按 episode 随机划分（会 leak 同场景）

Step 8 · Dataloader 验证
  用 lerobot-train / 官方 loader 跑 1 epoch
  可视化: 预测 action 叠加到图像上，肉眼看是否合理
```

---

## 七、三种典型训练 Recipe（对照表）

### Recipe 1 · OpenVLA 风格（开源可复现）

```
数据: OXE 970K episodes → Finetune 目标集 100-1K ep
模型: SigLIP + Llama2-7B + 离散 action head
Pretrain:
  - 冻结 SigLIP
  - LoRA on Llama
  - 全训 action head + projector
  - batch=256, lr=2e-5, 若干天 A100
Finetune:
  - LoRA rank=32
  - 更高 lr on action head
  - 数据增强: 颜色 jitter
Deploy:
  - 每 8 步 re-infer 一次
  - 7 维 delta EEF @ 5-10 Hz 推理
```

### Recipe 2 · π0 风格（工业精细操作）

```
数据: 大规模私有混合（估 10K+ 小时），多 embodiment
模型: PaliGemma + Flow Matching action expert
Pretrain:
  - Flow matching loss on action chunks (H=50)
  - 可能 joint train VLM 部分层
  - 连续动作，无 tokenization
Finetune:
  - 目标 task 50-200 demos 即可
  - 保持 flow head，调 denoising steps
Deploy:
  - 50 Hz 控制
  - Action chunk 执行前 k=25 步后 re-plan
```

### Recipe 3 · RDT 风格（中文社区 / 双臂）

```
数据: 自采 + OXE 子集 + Unified Action Space
模型: Diffusion Transformer (DiT) 1B
Pretrain:
  - 统一 128 维 action space
  - embodiment embedding
  - diffusion noise schedule
Finetune:
  - 换 embodiment id，少量 bimanual demo
Deploy:
  - 双臂 joint 或 EEF，取决于 unified space 解码
```

---

## 八、部署闭环：推理频率 vs 控制频率

```
┌──────────── 50 Hz 控制环 ────────────────────────────────────────┐
│                                                                  │
│  t=0        t=20ms      t=40ms      ...      t=200ms             │
│   │           │           │                    │                 │
│   ▼           ▼           ▼                    ▼                 │
│ 执行 chunk[0] 执行 [1]   执行 [2]   ...   执行 [9]               │
│                                           │                      │
│                                           ▼                      │
│                                    VLA 推理 (~100-200ms)         │
│                                    生成新 chunk[0:49]            │
└──────────────────────────────────────────────────────────────────┘
```

**关键参数**：

| 参数 | 典型值 | 说明 |
|------|--------|------|
| 控制频率 | 50 Hz | 低层 PD / IK 控制器 |
| 推理频率 | 5–10 Hz | GPU 跑一次 VLA |
| Chunk 长度 H | 25–100 | 推理一次管多少步 |
| 执行比例 k/H | 0.5–1.0 | 执行完整个 chunk 还是执行一半就 re-infer |
| 相机延迟 | < 30ms | 否则 closed-loop 不稳定 |

**ACT 的特殊性**：H 很大（100），但每步都条件于当前观测 → 介于开环 chunk 和闭环之间。

---

## 九、2026 共识：Data Pyramid 与负迁移

### Data Pyramid（训练数据配比）

```
                    ┌─────────────────┐
                    │ Tier 3: 精调    │  50-500 ep 目标机器人 teleop
                    │ 高质量真机      │  决定最终 success rate
                    ├─────────────────┤
                    │ Tier 2: 预训练  │  OXE + UMI + 仿真 MimicGen
                    │ 中质量混合      │  决定泛化能力
                    ├─────────────────┤
                    │ Tier 1: 先验    │  Ego 视频 + 互联网 + VLM co-train
                    │ 弱监督大规模    │  决定语义理解 breadth
                    └─────────────────┘
```

### 负迁移（Negative Transfer）

**现象**：加了某个数据集后，目标平台性能 **下降**。

**常见原因**：
- Action 空间不一致（joint vs EEF 混训未统一）
- 相机视角差异大（第三人称 vs wrist）
- 低质量数据未 QA 直接混入

**解法**：
- Unified Action Space（RDT）
- 按 embodiment 加权采样
- 先 Pretrain 广后 Finetune 窄，不要一步到位

---

## 十、和本地仓库的衔接

| 你想搞懂… | 本文章节 | 本地资源 | 算法论文 |
|----------|---------|---------|---------|
| 数据怎么采 | §五 | [论文索引 §1-5](../paper/论文索引.md) | — |
| 数据怎么处理 | §六 | [数据调研报告 §4](./机器人数据工作综合调研报告.md) | — |
| Pretrain 怎么训 | §二 Stage 1 | [RDT/Qwen 论文](../paper/论文索引.md#4-foundation) | [算法层清单](./VLA算法层学习路线与论文清单.md) |
| Action Head 区别 | §三 | [ACT 笔记](./paper-note/ACT-Model-Working-Principles.md) | Diffusion Policy, RT-2, OpenVLA |
| 部署闭环 | §八 | LeRobot deploy 文档 | π0, ACT |

**建议阅读顺序（深度版，3–4 周）**：

```
Week 1: 本文 §一–§四 + 快速入门 Day 0-5 + ACT 笔记
Week 2: 本文 §五–§六 + 数据调研报告 + UMI/ALOHA 笔记
Week 3: 算法层清单 Layer 1-4 + RT-2/OpenVLA 论文 Abstract
Week 4: 本文 §七–§九 + RDT/Qwen 论文 + 跑 LeRobot 一个 finetune demo
```

---

> **一句话总结**：现代 VLA = **冻结/LoRA 的 VLM 底座** + **Action Head（token 或 flow）** + **Data Pyramid（Ego→混合→真机精调）** + **八步数据流水线** + **Chunk 式闭环部署**。算法和数据是同一枚硬币的两面，缺任何一面都无法落地。
