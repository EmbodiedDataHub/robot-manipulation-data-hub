---
title: "Open-TeleVision: Teleoperation with Immersive Active Visual Feedback"
authors: "Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang, Xiaolong Wang"
year: 2024
source: "arXiv:2407.01512 · CoRL 2024"
tags:
  - Open-TeleVision
  - VR
  - teleoperation
  - humanoid
  - imitation-learning
  - ACT
  - active-perception
  - stereo-vision
aliases:
  - OTV
  - TeleVision
paper_path: "paper/Data Acquisition/VR Teleoperation/Open-TeleVision.pdf"
code_path: "https://github.com/OpenTeleVision/TeleVision"
project_website: "https://robot-tv.github.io"
related_notes:
  - "./VR与人形Teleop数采.md"
  - "../ALOHA/ACT-Model-Working-Principles.md"
  - "../ALOHA/ACTPolicy-Code-Walkthrough.md"
  - "../../数采到VLA训练-数据管线整体方案.md"
---

## 一句话总结

Open-TeleVision 提出 **沉浸式 VR 遥操作 + 机器人主动立体头显**：操作员在 VR 里看到机器人 **第一人称双目画面**（头随人转），手臂/灵巧手 motion retarget 到人形机器人，以 **60 Hz** 采集 demo 并训 **ACT（DinoV2 + stereo）**，在 H1/GR-1 上完成 **长时域精细任务**（分拣/插入/折毛巾/卸载），并支持 **跨洋远程** teleop——**感知创新 > 执行创新**。

---

## 论文基本信息

| 项 | 内容 |
|----|------|
| 标题 | Open-TeleVision: Teleoperation with Immersive Active Visual Feedback |
| 作者 | Xuxin Cheng\*¹, Jialong Li\*¹, Shiqi Yang¹, Ge Yang², Xiaolong Wang¹ |
| 机构 | UC San Diego¹ · MIT² |
| 发表 | arXiv:2407.01512 · **CoRL 2024** |
| 开源 | [robot-tv.github.io](https://robot-tv.github.io/) · [GitHub: OpenTeleVision/TeleVision](https://github.com/OpenTeleVision/TeleVision) |
| 本地 PDF | `paper/Data Acquisition/VR Teleoperation/Open-TeleVision.pdf` |
| 关联笔记 | [VR 数采概览](./VR与人形Teleop数采.md) · [ACT 原理](../ALOHA/ACT-Model-Working-Principles.md) |

---

## 1. 研究动机：现有 Teleop 的两块短板

Teleop 系统通常分 **执行（actuation）** 和 **感知（perception）** 两部分。论文认为：执行侧 VR+retarget 已可行，**感知侧才是精细人形操作的瓶颈**。

### 1.1 执行侧：各路线对比

| 路线 | 优点 | 局限 |
|------|------|------|
| **关节同构 copy**（ALOHA/GELLO） | 带宽高、精度高 | 需同地；难控多指手；每套机器人需专用 leader |
| **RGB 视觉 tracking**（AnyTeleop 等） | 便宜 | 噪声大、精度差 |
| **VR tracking**（本文） | 厂商融合多传感器，手/腕 pose 较稳 | 需 IK + retarget；引入额外误差 |

> 论文结论：**不用 joint copy，VR + hand retarget 也能做 ALOHA 级精细操作**——与 ALOHA 形成互补路线。

### 1.2 感知侧：遮挡与「远程 vs 深度」矛盾

| 感知方式 | 问题 |
|----------|------|
| **第三人称肉眼观察** | 臂/躯干遮挡；操作员必须站在 lab |
| **VR passthrough / 固定 RGB 流** | 远程可行，但 **失去立体深度**；边缘 PoI 需斜眼看，不直观 |
| **OPEN TEACH** | MR 合并 passthrough + 机器人相机，但 **远离机器人则深度感消失** |

**OTV 核心命题**（Appendix A）：

> 在 OTV 之前，没有系统能 **同时** 提供 **远程控制 + 深度感知**——要么到场肉眼立体看，要么看 RGB 流放弃深度。  
> **OTV 用机器人端 stereo streaming 打破这个互斥。**

### 1.3 论文要回答的问题

1. **主动头显相机**（随人头动）是否让 teleop 更直观、采到的数据更适合 IL？  
2. **立体视觉**对操作员和策略是否都关键？  
3. VR + retarget 能否支撑 **长 horizon、精细、双手** 任务？  
4. 能否 **跨网络远程** 采高质量 demo？

---

## 2. 系统总览（Fig.1）

```text
┌─────────────── 左：Teleoperation 采集 ───────────────┐
│  VR 设备 stream hand / head / wrist pose (SE(3))     │
│         ↓ Vuer Web Server                           │
│  retarget → robot joint position targets            │
│         ↓                                           │
│  机器人执行 + ZED Mini 回传 stereo RGB               │
└─────────────────────────────────────────────────────┘

┌─────────────── 右：Imitation Learning ───────────────┐
│  数据: stereo images + proprio + joint actions      │
│  算法: ACT (DinoV2 backbone, 2× stereo 输入)        │
│  输出: action chunk → 60Hz 闭环部署                 │
└─────────────────────────────────────────────────────┘
```

**整环频率**：**60 Hz**（pose 上行 + stereo 下行 + joint command）。

---

## 3. 硬件设计（Fig.3）

### 3.1 两套人形平台

| | **Unitree H1** | **Fourier GR-1** |
|---|----------------|------------------|
| 臂 | 7DoF × 2 | 7DoF × 2 |
| 末端 | **Inspire 6DoF 灵巧手** × 2（12 DoF/手，6 驱动） | **1DoF jaw 平行夹爪** × 2 |
| 主动颈 | **自制 2DoF gimbal**（yaw+pitch），3D 打印 + DYNAMIXEL XL330 | **原厂 3DoF 颈**（yaw+roll+pitch） |
| 相机 | **ZED Mini** stereo RGB | 同左 |
| 使用 DoF | 颈+双臂+手；**腿/躯干未用** | 同左 |

**为何做人形**：遮挡与直觉性问题在 **人形 + 高 DoF** 上最突出；系统也适用于「双臂+一相机」任意配置。

### 3.2 主动感知（Active Sensing）

- H1：2DoF gimbal 装在 torso 顶部，模拟人颈 **yaw/pitch**  
- GR-1：直接用厂商 3DoF 颈  
-  teleop 时：**相机随操作员头动**，流式传输 **egocentric stereo** 到 VR  

与 **静态广角相机** 对比（Fig.6）：单相机难以同时覆盖所有 PoI；多相机需 per-task 调位。主动相机 = **自然注意力机制**，训练/推理 token 更少（Tab.2：训练 **2× 快**，部署 **83Hz vs 42Hz**）。

---

## 4. 软件栈与通信

| 组件 | 作用 |
|------|------|
| **Vuer** [21] | Web 服务器；VR ↔ 机器人桥接 |
| **Apple Vision Pro** | 论文主 VR；系统声明 **device-agnostic**（Appendix D 亦支持 Quest / 平板） |
| **Pinocchio** | FK/IK、SE(3) 插值 |
| **dex-retargeting** | 人手 → 灵巧手/夹爪 joint |
| **NLopt SLSQP** | retarget 优化求解 |

**分辨率**：每眼 **480×640** RGB stereo stream。

**远程**：经 internet（论文案例 MIT Boston → UCSD San Diego，~3000 miles）；GitHub 支持 `ngrok=True`。

---

## 5. 执行层详解

### 5.1 臂控制：CLIK + 相对头部位姿

**输入**：VR 估计的人 **腕部 SE(3)**。

**映射规则**（关键设计）：

| 量 | 规则 | 目的 |
|----|------|------|
| **EEF 位置** | 人腕相对 **头** 的位置 → 机器人 EEF 相对 **机器人头** | 人头动时，手在空间中相对稳定 |
| **EEF 朝向** | 人腕 **绝对朝向** → 机器人腕绝对朝向 | 保持抓取姿态 |

**IK**：

- **CLIK**（Closed-loop IK，Pinocchio）  
- SE(3) **group filter** 平滑输入  
- 可操作度接近极限 → Jacobian **零空间 joint offset**，减少 IK 失败且尽量不破坏 tracking  

### 5.2 手控制：dex-retargeting 向量优化

优化目标（论文 Eq.1）：

$$\min_{q_t} \sum_i \|\alpha v_t^i - f_i(q_t)\|^2 + \beta \|q_t - q_{t-1}\|^2$$

- $v_t^i$：人 hand keypoint 向量  
- $f_i(q_t)$：机器人 hand FK 对应向量  
- $\alpha = 1.1$：人手 vs Inspire 尺寸缩放  
- $\beta$：时间一致性  

**灵巧手（H1）**：7 条向量——腕→五指尖（5）+ 拇指尖→食/中指尖（2）。  

**夹爪（GR-1）**：1 条向量——人拇食间距 ↔ 夹爪上下颚间距（捏合控制开合）。

### 5.3 Inspire 手结构（Appendix C）

- 每手 5 指，**12 DoF**（**6 驱动 + 6 欠驱动连杆**）  
- 四指：MCP 单驱动 → PIP 连杆跟随  
- 拇指：CMC 双驱动 + MCP/IP 连杆  

---

## 6. 感知层：为何 Stereo + Active 是论文主贡献

### 6.1 对 Teleop 操作员

- **主动相机**：PoI 可保持在 **中央凹视觉（foveal）** 区域，无需斜眼看图像边缘  
- **Stereo vs Mono**（User Study, Tab.3, 4 人）：

| 指标 | Stereo | Mono（仅左眼 RGB） |
|------|:------:|:----------------:|
| 任务成功率均值 | **100%** | 71–93%（Unloading 仅 **50%**） |
| 完成时间 | **更短**（如 Sorting 66s vs 91s） |

定性反馈：远程 teleop 仅靠单目 RGB，**深度直觉不足**；stereo 显著减轻不适。

### 6.2 对 Imitation Learning 策略

| 消融 | Can Sorting pick (H1) | 说明 |
|------|:---------------------:|------|
| **DinoV2 + Stereo（Ours）** | **92%** | 完整系统 |
| w. ResNet18 | 74% | 缺 spatial 表达 |
| w/o Stereo Input | **46%** | 几乎无法可靠抓取 |

**论点**：策略不仅学 manipulation，还学 **何时/如何转头**——主动相机轨迹是 demo 的一部分。

---

## 7. 数据采集

### 7.1 每步记录

```text
observation:
  · stereo_left, stereo_right (RGB)
  · proprio: 当前全部 controlled joint 读数

action (BC 标签):
  · 绝对 joint position targets（与部署一致）
  · H1: 28D = 7×2 arm + 6×2 hand + 2 neck
  · GR-1: 19D = 7×2 arm + 1×2 gripper + 3 neck

language:
  · 任务级自然语言（episode 级）

episode 边界:
  · 操作员做固定「结束手势」→ 停止录制
  · 结束手势保留在数据中（策略可学终止条件）
```

### 7.2 任务与 demo 规模（Appendix E, Tab.6）

| 任务 | 平均 episode 时长 | Demo 条数 | 备注 |
|------|:-----------------:|:---------:|------|
| H1 Can Sorting | 93±5 s | **10** | 每条含 **10 次子分拣** → 等效 100 rollouts |
| GR-1 Can Sorting | 61±5 s | **10** | 每条 **6 次** |
| Can Insertion | 84±7 s | 20 | 每条 6 罐插入 |
| Folding | 44±5 s | 20 | 完整折两次 |
| Unloading | 93±6 s | 20 | 4 tube × 提取+传递+放置 |

**Can Sorting demo 少但子任务多**：重复性高，10 条 episode 已含足够子轨迹。

### 7.3 四类任务设计要点

| 任务 | 考察能力 | 特殊策略 |
|------|----------|----------|
| **Can Sorting** | 位置/颜色 OOD 泛化；需 **主动转头** 看左右 bin | 全掌五指抓+投掷式放置 |
| **Can Insertion** | **亚厘米级**插入（罐 Ø5.6cm → 孔 Ø7.6cm） | **拇+食二指捏**；chunk=**100** |
| **Folding** | **软体**、力控、双手协调 | 重复性高，各 baseline 均 100% fold |
| **Unloading** | 视觉推理 tube 在哪 + **双手传递** | 四 slot 随机 |

---

## 8. 模仿学习：ACT 变体

### 8.1 相对 ALOHA/ACT 的改动

| 项 | 原版 ACT (ALOHA) | Open-TeleVision |
|----|------------------|-----------------|
| 视觉 backbone | ResNet18 | **DinoV2-ViT** |
| 输入相机 | 4 路独立 RGB | **2 路 stereo**（左+右） |
| Image tokens | ResNet 特征 | 每图 **16×22** tokens |
| Action 空间 | 14D joint | **28D (H1) / 19D (GR-1)** 绝对 joint |
| Chunk size | 100 | **60**（Insertion 用 **100**） |
| 控制频率 | ~50 Hz | **60 Hz** |

**为何不用 Diffusion**：论文走 **teleop→ACT** 成熟管线；重点在 **数据质量**，非 action head 创新。

### 8.2 训练超参（Tab.7）

| 超参 | 值 |
|------|-----|
| KL weight | 10 |
| chunk size | 60（Insertion: 100） |
| hidden dim | 512 |
| batch size | 45 |
| feedforward dim | 3200 |
| epochs | 25000 |
| learning rate | 5e-5（AdamW） |
| temporal weighting $m$ | 0.01（Sorting: 0.005；Unloading: 0.05） |
| GPU | 单卡 **RTX 4090** |

**Temporal aggregation**：沿用 ACT 指数加权 $w_i = \exp(-m \cdot i)$；$m$ 小 → 更 react；$m$ 大 → 更 steady。

### 8.3 主要结果（Tab.1 摘要）

**H1 Can Sorting**：Ours pick **92%** / place **88%**；无 stereo pick 仅 **46%**。  

**Can Insertion**：Ours pick **90%** / insert **87%**；无 stereo 约 **47%/63%**。  

**Folding / Unloading (H1)**：Ours 多项 **100%**；ResNet18 在 Unloading extract 有失败。  

**GR-1 Can Sorting place 偏低（60%）**：夹爪抓罐时 **遮挡颜色** → Appendix B 贴标签后 place **→100%**。

---

## 9. 泛化实验（Fig.5, Fig.9）

- **H1 Can Sorting**：4×4 网格（格 3cm），训练区 **100%** pick；边缘少见位点仍有适应  
- **GR-1 + 贴标 cans**：网格上几乎 **全 100%**（gripper 抓罐比灵巧手更宽容）  

---

## 10. 扩展 Teleop 能力（Fig.7）

| 任务 | 说明 |
|------|------|
| **Wood-board Drilling** | 左手持板 + 右手电钻；**食指控扳机**；需灵巧手（夹爪几乎不可能） |
| **Earplugs Packing** | 随机 earplug → 随机 latch box；双手 in-hand 调整 |
| **Pipette Transfer** | 拇指 DoF 按 pipette；tube 直径 **1.5cm** 级精度 |

说明：即使用 **准直驱 + 行星减速**（齿隙）的 H1，有人 in the loop 仍可做高精度 teleop。

---

## 11. 与 prior work 对照（Tab.4）

| 系统 | Actuation | 多指手 | 双手 | Perception | 远程 | 深度 |
|------|-----------|:------:|:----:|------------|:----:|:----:|
| OPEN TEACH | VR Controller | ✓ | ✓ | Direct+RGB | ✗ | ✓ |
| ALOHA | Joint Copy | ✗ | ✓ | Direct View | ✗ | ✓ |
| GELLO | Joint Copy | ✗ | ✓ | Direct View | ✗ | ✓ |
| AnyTeleop | RGB Tracking | ✓ | ✗ | Direct/RGB | ✓ | ✓ |
| **Open-TeleVision** | VR Tracking | ✓ | ✓ | **Stereo Active** | **✓** | **✓** |

**独特组合**：VR tracking + 多指双手 + **第一人称主动 stereo** + **远程**。

---

## 12. 局限与未来（§5）

| 局限 | 说明 |
|------|------|
| **无触觉反馈** | 精细接触任务仍主要靠视觉 |
| **无在线 relabel / 纠错** | 失败 demo 需重录 |
| **mobility 未用** | 仅 upper body + neck；未来可扩展到 mobile |
| **IK/retarget 失败** | 需 QA 过滤；不如 joint copy 稳 |
| **设备成本** | Vision Pro + 人形平台远高于 ALOHA |

---

## 13. 在数采→训练管线中的位置

```text
范式 C · VR 人形 Teleop
  ↓
Raw: stereo MP4 + joint state @ 60Hz
  ↓ 同步 / QA(IK fail, blur)
Canonical IR: images + joint_pos_target + language
  ↓
Export: HDF5 / LeRobot
  ↓
Train: ACT (DinoV2, chunk=60)
  ↓
Deploy: 60Hz chunk + temporal aggregation
```

详见 [数采总方案 §6.3](../../数采到VLA训练-数据管线整体方案.md) · [VR 数采概览](./VR与人形Teleop数采.md)

**与 UMI / ALOHA 动作空间**：

| 系统 | Action | 需 IK 部署 |
|------|--------|:----------:|
| ALOHA | 14D joint | ❌ |
| OTV | 28/19D joint | ❌（策略直接出 joint） |
| UMI | relative EEF | ✅ |

---

## 14. 视频 · 代码 · 推荐阅读

### 14.1 视频（建议顺序）

| # | 资源 | 看什么 |
|:-:|------|--------|
| 1 | [robot-tv.github.io](https://robot-tv.github.io/) · **Autonomous Skills** | 训完后的长 horizon 自主执行 |
| 2 | 同页 · **Cross-Country Teleoperation** | Boston↔San Diego 远程 |
| 3 | 同页 · **Teleoperation** | 钻孔/包装/移液等扩展 |
| 4 | [The Robot Report 报道](https://www.therobotreport.com/mit-uc-san-diego-researchers-create-open-television-immersive-teleoperation/) | 5 min 读懂动机 |

### 14.2 代码

```bash
git clone https://github.com/OpenTeleVision/TeleVision
conda create -n tv python=3.8 && conda activate tv
pip install -r requirements.txt
cd act/detr && pip install -e .

# Teleop
cd teleop && python teleop_hand.py

# Train
python imitate_episodes.py --policy_class ACT --kl_weight 10 \
  --chunk_size 60 --hidden_dim 512 --batch_size 45 \
  --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5
```

### 14.3 论文精读顺序

1. **§1 Introduction** — 动机与互斥矛盾  
2. **§2 TeleVision System** — 60Hz 闭环 + 硬件  
3. **§2 Arm/Hand Control** — IK 与 retarget 细节  
4. **§3 Experiments** — 四任务 + 消融  
5. **Appendix A** — 与 OPEN TEACH/ALOHA 感知对比  
6. **Appendix E** — demo 规模与超参  
7. 对照 [ACT 原理](../ALOHA/ACT-Model-Working-Principles.md) 理解 chunk / CVAE / temporal agg  

---

## 15. 自测

1. OTV 相对 ALOHA 的核心差异在 **执行** 还是 **感知**？（主要是 **感知**：主动 stereo；执行走 VR+IK 而非 joint copy）  
2. 为何 EEF 位置要相对 **头** 映射？（头动时保持手相对空间稳定）  
3. 为何 Can Sorting 只采 **10 条** episode 仍够训？（每条含 10/6 次子任务）  
4. GR-1 place 成功率低的主因？（夹爪遮挡罐体颜色 → 视觉无法辨色）  
5. 训练时 stereo 为何重要？（单目缺 depth → pick 46%）  

---

## 16. 关联笔记

| 文档 | 关系 |
|------|------|
| [VR 与人形 Teleop 数采](./VR与人形Teleop数采.md) | 三篇 VR 论文对比 + 学习路径 |
| [ALOHA 论文笔记](../ALOHA/ALOHA-Learning-Fine-Grained-Bimanual-Manipulation.md) | joint copy 对照 |
| [ACT 原理](../ALOHA/ACT-Model-Working-Principles.md) | 算法细节 |
| [OPEN TEACH](./VR与人形Teleop数采.md#三open-teach--低成本通用-vr-teleop) | 低成本 VR 对照 |
| [IL 范式 · ACT](../IL-Paradigms/概述.md) | 在 IL 地图中的位置 |
