---
title: "ALOHA: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
authors: "Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn"
year: 2023
source: "arXiv:2304.13705"
tags:
  - ALOHA
  - ACT
  - teleoperation
  - imitation-learning
  - bimanual-manipulation
  - fine-grained-manipulation
  - low-cost-robotics
aliases:
  - ALOHA 2023
pdf_path: "paper/Data Acquisition/Robot Teleoperation/ALOHA - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.pdf"
code_path: "code/act"
project_website: "https://tonyzhaozh.github.io/aloha"
---

## 一句话总结

ALOHA 用 **<$20k 的双臂关节空间遥操作硬件** 采集高质量 demo，配合 **ACT（Action Chunking with Transformers）** 这一 CVAE + Transformer 模仿学习算法，在真实世界中仅用 **约 10 分钟 demo** 就能学会开透明酱料杯、插电池等 **毫米级精度** 的双臂精细操作，成功率 **80–90%**。

---

## 论文基本信息

| 项 | 内容 |
|----|------|
| 标题 | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware |
| 作者 | Tony Z. Zhao¹, Vikash Kumar³, Sergey Levine², Chelsea Finn¹ |
| 机构 | Stanford¹, UC Berkeley², Meta³ |
| 发表 | arXiv:2304.13705, 2023-04-23 |
| 开源 | 硬件 + 软件 + 教程全部开源 |
| 代码 | [tonyzhaozh/aloha](https://github.com/tonyzhaozh/aloha) → 本仓库 `code/act` |

---

## 1. 研究动机：为什么要做这件事

### 1.1 精细操作为什么难

论文关注的不是「抓放」，而是 **fine-grained manipulation（精细操作）**：

- 开透明酱料杯盖：需要先拨倒、再轻推入另一夹爪、再撬盖
- 插电池：需要对准槽位、对抗弹簧反力、双手协调固定遥控器
- 穿 Velcro 扎带：loop 仅 3mm×25mm，毫米级误差即失败

这类任务共同特点：

1. **高精度**：毫米级误差导致失败
2. **富接触**： pinch / pry / tear，不是自由空间运动
3. **强闭环**：必须依赖视觉反馈持续修正
4. **双手协调**：左右臂分工、递接、对抗力

### 1.2 现有方案的痛点

| 路线 | 问题 |
|------|------|
| 高端机器人 + 精确传感器 + 标定 | 贵、难复现、实验室门槛高 |
| 纯模型 / 规划 | 透明物体、软体变形、复杂接触建模极难 |
| 普通 BC 模仿学习 | compounding error；人类 demo 有 pause、多模态 |

### 1.3 论文的核心问题

> **能否用低成本、不精确的硬件，通过端到端学习，完成通常需要高端平台才能做的精细操作？**

论文答案：**可以** —— 关键是 **高质量 teleop 数据** + **专门为精细操作设计的 IL 算法（ACT）** 的协同。

---

## 2. 整体贡献：两大组件

```
┌─────────────────────────────────────────────────────────────┐
│                    ALOHA 完整系统                             │
├──────────────────────────┬──────────────────────────────────┤
│  ALOHA 硬件（数据引擎）    │  ACT 算法（策略学习）              │
│  · <$20k 双臂 teleop      │  · Action Chunking               │
│  · 关节空间 leader-follower│  · Temporal Ensembling           │
│  · 4 相机 @ 50Hz          │  · CVAE 建模人类多模态 demo       │
│  · 10min demo → 80-90% SR │  · Transformer 架构               │
└──────────────────────────┴──────────────────────────────────┘
```

**核心论点**：硬件负责「采到好数据」，算法负责「从 noisy、非 Markov 的人类 demo 中学到精确闭环策略」。二者缺一不可。

---

## 3. ALOHA 硬件系统精读

### 3.1 五条设计原则

1. **Low-cost**：预算与单条工业臂相当（~$20k 整机）
2. **Versatile**：真实世界多种精细任务
3. **User-friendly**：直观、可靠、易用
4. **Repairable**：Dynamixel 电机可快速更换
5. **Easy-to-build**：非专家 **<2 小时** 可组装

### 3.2 机器人选型

| 角色 | 型号 | 价格 | 规格 |
|------|------|------|------|
| **Follower（执行）** | ViperX 300 6-DoF ×2 | ~$5600/臂 | 负载 750g，臂展 1.5m，重复精度 1mm，**精度 5–8mm** |
| **Leader（遥操作）** | WidowX 250 6-DoF ×2 | ~$3300/臂 | 与 ViperX 同厂、近似缩放版 |

- **不用灵巧手**：维护贵、复杂度高
- **自制 3D 打印透明夹爪 + 防滑胶带**：薄边、可视性好，能抓塑料膜

### 3.3 为什么用关节空间映射，而不是 VR / 任务空间

Leader 关节角 **直接同步** 到 Follower（joint-space mapping），而非 VR 手柄 → IK → EE pose。

| 优势 | 原因 |
|------|------|
| 避免 IK 奇异 | 6-DoF 无冗余，精细操作常在奇异附近，现成 IK 频繁失败 |
| 低延迟 | 关节映射计算量小 |
| 天然限速 / 阻尼 | Leader 重量限制操作者过快移动，振动更小 |
| 操作感更好 | 论文称 VR 控制器在精确任务上不如 backdrive |

### 3.4 人机工程学改造

- **Handle & Scissor 机构**（3D 打印）：降低 backdrive 力，夹爪 **连续** 控制（非开/关二值）
- **橡皮筋负载平衡**：抵消 Leader 重力，支持 **>30 分钟** 连续 teleop
- **笼式框架**：20×20mm 铝型材 + 交叉钢缆

### 3.5  sensing 与数据频率

| 传感器 | 规格 |
|--------|------|
| 相机 ×4 | Logitech C922x，480×640 RGB，30fps 流 |
| 视角 | 顶视、前视（旋转 90° 增垂直视野）、双腕视 |
| 控制 / 录制频率 | **50Hz** |
| 动作定义 | **Leader 关节位置**（不是 Follower！） |

**为何录 Leader 而非 Follower**：两者差值经底层 PID 体现 **施加力**，Leader 位置隐式编码了操作者的 force intent。

### 3.6 成本对比（Appendix A）

| 系统 | 估计成本 | 能力 |
|------|---------|------|
| DexPilot（KUKA + Allegro） | ~$100k（单臂+手） | 钱包取卡、NIST 插装 |
| Shadow Teleoperation（双臂灵巧手+UR10×2） | **>$400k** | 最强，15 项 demo 中 ALOHA 复现 14 项 |
| Robotic Telekinesis 等 | ~$18k（单臂+手） | 单 RGB 手姿 retarget |
| **ALOHA** | **~$18–20k** | 穿扎带、插 RAM、颠乒乓球等 |

论文强调：平行夹爪换灵巧手 → 轻量低成本臂 → 更 nimble、少维护。

### 3.7 ALOHA 能 teleop 的任务类型

- **Precise**：穿 zip tie、从钱包取卡、开 ziploc
- **Contact-rich**：插 288-pin RAM、翻书页、NIST board #2 装链
- **Dynamic**：乒乓球颠球、空中甩开口袋

---

## 4. ACT 算法精读

### 4.1 要解决的 IL 难题

1. **Compounding error**：单步 BC 小误差 → 状态 OOD → 不可恢复
2. **Non-Markovian demo**：人类演示中的 **pause**、节奏变化，单步 Markov 策略难建模
3. **Human multi-modality**：同一观测多种合理轨迹（如 mid-air 递接位置每次不同）

### 4.2 Action Chunking（动作分块）

**灵感**：神经科学 action chunking —— 把动作序列分组为「块」一次执行。

**实现**：

- 固定 chunk 大小 **k**（默认 **k=100**）
- 策略建模 **π(a_{t:t+k} | s_t)**，而非 π(a_t | s_t)
- 有效 horizon 缩短 **k 倍** → 减轻 compounding error
- 块内 pause 等时间相关 confounder 可被 chunk 吸收

**执行方式（两种）**：

| 模式 | 做法 |
|------|------|
| 标准 chunking | 每 k 步观测一次，顺序执行 k 步动作 |
| **Temporal Ensembling（TE）** | **每步都 query**，对「同一时刻 t 的多个历史预测」做指数加权平均 |

TE 权重：\( w_i = \exp(-m \cdot i) \)，\(w_0\) 为最旧预测；\(m\) 越小，新观测融入越快。

> TE 的关键：聚合的是 **同一 timestep 的多个预测**，不是相邻 timestep 的动作（避免 bias）。

### 4.3 CVAE：建模人类 demo 的多模态性

ACT 训练为 **Conditional VAE**：

```
训练：
  Encoder q_φ(z | a_{t:t+k}, qpos)     ← 故意不用图像，加速训练
  Decoder π_θ(â_{t:t+k} | o_t, z)      ← o_t = 图像 + 关节

推理：
  丢弃 Encoder
  z = 0（先验均值）
  π_θ(â_{t:t+k} | o_t, z=0)
```

| 设计 | 原因 |
|------|------|
| Encoder 不看图像 | 训练更快；z 从 **动作序列 + qpos** 编码「风格/意图」 |
| z ~ N(0,I) 先验 + β 加权 KL | 推理时 z=0 可用；β=10（Table III） |
| L1 而非 L2 重建 | 动作序列建模更精确 |
| **绝对关节位置** 而非 delta | delta 性能显著下降 |

### 4.4 网络架构（Fig. 4, 11 + Appendix C）

**参数量 ~80M**，每任务 from scratch 训练，RTX 2080 Ti 约 **5 小时**，推理 **~0.01s**。

#### CVAE Encoder（BERT-like Transformer Encoder）

```
输入序列 (k+2 个 token):
  [CLS] + embedded_qpos + embedded_action_seq(k步)
  每个 action: Linear(14 → 512)
  加正弦位置编码
  → 4 层 Transformer Encoder
  → 取 [CLS] 输出 → Linear(512 → 64) → μ(32), logvar(32)
  → reparameterize → z(32) → Linear(32 → 512)
```

#### CVAE Decoder / Policy

```
4 × RGB 480×640×3
  → ResNet18 → 15×20×512 feature map / 相机
  → flatten → 300×512 / 相机
  → 4 相机拼接 → 1200×512
  → 追加 z(512) + qpos(512) → 1202×512 memory tokens

Transformer Encoder（4层）编码 memory
Transformer Decoder（7层）：
  query = k 个固定位置编码 (k×512)
  cross-attend memory
  → MLP → k×14 绝对关节目标
```

**超参（Table III）**：

| 参数 | 值 |
|------|-----|
| chunk size k | 100 |
| hidden dim | 512 |
| encoder layers | 4 |
| decoder layers | 7 |
| heads | 8 |
| FFN dim | 3200 |
| β (KL weight) | 10 |
| lr | 1e-5 |
| batch size | 8 |

### 4.5 训练 / 推理伪算法（论文 Algorithm 1 & 2）

**训练**：

1. 从 demo 采样 (o_t, a_{t:t+k})
2. z ~ q_φ(z | a_{t:t+k}, qpos)
3. â_{t:t+k} ~ π_θ(· | o_t, z)
4. L = MSE/L1(â, a) + β · D_KL(q_φ || N(0,I))

**推理**：

1. 每步（或每 k 步）用 z=0 预测 â_{t:t+k}
2. TE 模式下写入 FIFO buffer B[t…t+k]
3. 对 B[t] 中所有预测加权平均 → 执行 a_t

---

## 5. 实验设计精读

### 5.1 8 个任务

**6 个真实任务**（ALOHA 硬件，50 demo/任务，~10–20 分钟数据）：

| 任务 | 难点 | 子任务 |
|------|------|--------|
| **Slide Ziploc** | 透明袋、变形随机 | Grasp → Pinch → Open |
| **Slot Battery** | 弹簧反力、需推边 | Grasp → Place → Insert |
| **Open Cup** | 小杯、透明、撬盖 | Tip Over → Grasp → Open Lid |
| **Thread Velcro** | 3mm loop、低对比度 | Lift → Grasp → Insert |
| **Prep Tape** | mid-air 递接 | Grasp → Cut → Handover → Hang |
| **Put On Shoe** | 紧配合、摩擦 | Lift → Insert → Support → Secure |

**2 个仿真任务**（MuJoCo）：Transfer Cube、Bimanual Insertion（间隙 ~1cm / ~5mm）

**感知挑战**：透明/半透明物体（袋、杯、胶带）、黑色桌面对黑色扎带低对比、腕视中小目标。

### 5.2 数据采集细节

| 项 | 数值 |
|----|------|
| 控制频率 | 50Hz |
| 每 episode 时长 | 8–14 秒（400–700 步） |
| demo 数量 | 50/任务（Thread Velcro 100） |
| 采集 wall-clock | 30–60 分钟/任务（含 reset 和失误） |
| 随机化 | 物体沿 15cm 白线随机；ziploc 从 ~5cm 高度落下 |

**人类 demo 的随机性**：同一任务不同 episode 轨迹不同（如 tape handover 位置），策略须学 **约束**（不碰撞、可抓）而非死记坐标。

### 5.3 Baseline

| 方法 | 特点 |
|------|------|
| **BC-ConvMLP** | 图像 CNN + 关节 → 单步动作 |
| **BeT** | Transformer + 离散化动作 + history；图像用 **冻结** 预训练 encoder |
| **RT-1** | Transformer + 离散 256 bins + 6 步 history |
| **VINN** | 非参数，测试时检索 kNN demo |
| **ACT** | chunk + CVAE + 连续动作 + 联合训练视觉 |

---

## 6. 实验结果精读

### 6.1 主结果（Table I & II）

**Table I**（2 sim + 2 real，ACT 全面碾压）：

| 任务 | ACT | 次优 |
|------|-----|------|
| Cube Transfer (sim, human) | **90%** | BeT 51% |
| Bimanual Insertion (sim, human) | **60%** | BeT 13% |
| Slide Ziploc (real) | **88%** | BeT 0% |
| Slot Battery (real) | **96%** | BeT 0% |

**Table II**（剩余 3 个 real 任务）：

| 任务 | ACT 最终成功率 | BeT |
|------|--------------|-----|
| Open Cup | **84%** | 0% |
| Thread Velcro | **20%** | 0% |
| Prep Tape | **64%** | 0% |
| Put On Shoe | **92%** | 0% |

**Baseline 失败模式**：

- 前几子任务偶尔成功，**最终成功率 <30%**
- episode 末尾性能崩溃、**无限 pause**
- 人类 demo 比 scripted 数据 **显著更难**（随机性 + 多模态）

**Thread Velcro 低成功率原因**（92%→40%→20% 逐级腰斩）：

1. 右夹爪过早闭合，mid-air 抓尾失败
2. 插入 loop 精度不足
3. 黑色扎带 + 小像素占比 → 视觉难

### 6.2 消融实验（Fig. 8）

#### (a) Chunk size k

| k | 平均成功率（ACT 无 TE） |
|---|------------------------|
| 1（无 chunking） | **1%** |
| 100 | **44%** |
| 200–400（近开环） | 略降（41–42%） |

→ chunking 是 **最关键** 设计；k 过大失去 reactive 能力。

对 BC-ConvMLP、VINN 加 chunking 同样 **大幅提升** → chunking 是 **通用技巧**。

#### (b) Temporal Ensembling

| 方法 | 无 TE | 有 TE | Δ |
|------|-------|-------|---|
| ACT | 44% | 47.3% | +3.3% |
| BC-ConvMLP | 25% | 29% | +4% |
| VINN | 37% | 17% | **-20%** |

→ TE 平滑参数化模型的误差；VINN 检索 GT 动作，TE 反而有害。

#### (c) CVAE

| 数据 | With CVAE | No CVAE |
|------|-----------|---------|
| Scripted | 59% | 58% |
| Human | **35.3%** | **2%** |

→ CVAE 对 **人类 demo 至关重要**；scripted 数据几乎不需要。

#### (d) 50Hz vs 5Hz Teleop 用户实验

6 名参与者，穿 zip tie + 拆塑料杯：

| 任务 | 5Hz | 50Hz | 变化 |
|------|-----|------|------|
| 穿 zip tie | 33s | 20s | -39% |
| 拆杯 | 16s | 10s | -38% |

→ 5Hz 导致 **62%** 完成时间增加，p < 0.001。

**结论**：精细操作 **必须高频率闭环**，不能用 RT-1 等低频 IL 的假设。

---

## 7. 失败案例与局限（Appendix F）

### 7.1 硬件局限

- 需 **多指配合** 的任务（儿童安全药瓶）
- **大力** 任务（拧 sealed 瓶、重物体）
- 需 **指甲/薄边** 操作（胶带自粘、拉环）

### 7.2 算法 / 数据局限

| 任务 | 结果 | 原因 |
|------|------|------|
| **Unwrap Candy** | 拿起 10/10，拉 8/10，剥开 0/10（10 trial）；5 糖各 10 trial 则 3/5 成功 | 接缝位置随机、视觉难辨 |
| **平放 ziploc 开袋** | 可拿起，后续 mid-air 步骤失败 | 感知难 +  pickup 位置敏感导致大变形 |

### 7.3 论文 Conclusion 中的定位

- 80–90% 成功率 @ ~10min demo 是 **真实世界 direct IL** 的强结果
- 仍有任务超出能力（如 **扣衬衫纽扣**）
- 希望成为 **accessible resource** 推动精细操作研究

---

## 8. 与代码实现的对应

本仓库 `code/act/` 即论文 ACT 官方实现，与论文差异点：

| 论文 | 代码 |
|------|------|
| MSE 重建（Algorithm 1） | 实际用 **L1**（`policy.py`，论文 §4 C 也提到 L1 更好） |
| TE 默认使用 | 代码需 `--temporal_agg` 才开启 |
| 标准 chunking | 默认 `query_frequency = num_queries = 100` |

详细代码导读见同目录 [`ACT-Model-Working-Principles.md`](ACT-Model-Working-Principles.md)。

---

## 9. 精读要点：这篇论文真正新在哪里

### 9.1 不是单点创新，是系统协同

| 层面 | 贡献 |
|------|------|
| **数据** | 证明 $20k 平台 + 50Hz joint teleop 能采到 **SOTA 级精细 demo** |
| **算法** | chunking + CVAE + TE 组合，专门打 compounding error + human noise |
| **评估** | 8 个 **真正难** 的双臂任务，不是 pick-place |

### 9.2 三个设计选择的因果链

```
50Hz teleop ──→ 高保真 demo
       │
       ▼
Action Chunking ──→ horizon ÷ k ──→ 少 compounding error
       │
       ▼
CVAE ──→ 建模 human multi-modality ──→ 人类 demo 可学（35% vs 2%）
       │
       ▼
Temporal Ensemble ──→ 平滑 chunk 边界 ──→ 精确 + 流畅
```

### 9.3 对后续工作的启示

1. **数据质量 > 模型容量**：BeT/RT-1 有 Transformer 但无 chunking → 精细任务 0%
2. **连续动作 > 离散化**：精细操作需要连续 offset，256 bins 不够
3. **联合训练视觉**：BeT 冻结 encoder → 感知-控制不匹配
4. **频率是硬约束**：5Hz vs 50Hz 差 62% 效率
5. **10 分钟 demo 够吗？** 对 6 个任务够；unwrap candy / flat ziploc 不够 → 需更多数据或预训练

---

## 10. 关键数字速查

| 指标 | 数值 |
|------|------|
| 系统总成本 | <$20k |
| 控制 / 录制频率 | 50Hz |
| Demo 量 | ~50 episodes ≈ 10min / 任务 |
| Chunk size k | 100 |
| Latent dim | 32 |
| 模型参数 | ~80M |
| 训练时间 | ~5h / 1×2080Ti |
| 推理延迟 | ~0.01s |
| 真实任务成功率 | 80–96%（除 Thread Velcro 20%） |
| KL weight β | 10 |

---

## 11. 推荐阅读顺序

1. **Abstract + Fig.1**：建立「低成本 + 精细任务」直觉
2. **§III ALOHA 硬件**：理解数据从哪来、为何 50Hz / joint mapping
3. **§IV ACT + Fig.4/5**：算法三部分（chunk / CVAE / TE）
4. **§V 实验 + Table I/II**：看 baseline 为何全灭
5. **§VI 消融**：确认每个组件的独立贡献
6. **Appendix C/F + 代码 `code/act`**：架构细节与实现

---

## 12. 与其他 ALOHA 系列论文的关系

| 论文 | 关系 |
|------|------|
| **ALOHA 2** | 硬件升级（更高保真 teleop） |
| **Mobile ALOHA** | 同 ACT，加速移动底座 + whole-body teleop |
| **本论文** | 原始系统：固定底座双臂 + ACT 算法 debut |

---

## 参考文献（论文核心引用）

- Action chunking 心理学背景：Lai et al. 2022
- CVAE：Sohn et al. 2015; Higgins β-VAE 2016
- Transformer / DETR：Vaswani 2017; Carion 2020
- Compounding error：Ross et al. 2010 (DAgger)
- Temporal confounders：Swamy et al. 2022
