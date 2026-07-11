---
title: "VR 与人形 Teleop 数采"
tags:
  - VR
  - teleoperation
  - humanoid
  - data-collection
  - Open-TeleVision
  - OPEN-TEACH
  - TWIST2
related_notes:
  - "../../数采到VLA训练-数据管线整体方案.md"
  - "../ALOHA/概述.md"
  - "../ALOHA/ACT-Model-Working-Principles.md"
pdf_path:
  - "paper/Data-Collection/VR-Humanoid/Open-TeleVision.pdf"
  - "paper/Data-Collection/VR-Humanoid/OPEN TEACH.pdf"
  - "paper/Data-Collection/VR-Humanoid/TWIST2- Scalable, Portable, and Holistic Humanoid Data Collection System.pdf"
updated: "2026-07-10"
---

# VR 与人形 Teleop 数采 · 详解

> **范式 C**：用 VR 捕获人体 pose → IK/retarget → 机器人 joint，同步录制 **头戴/主动相机** 与 whole-body 状态。  
> 本仓库三篇核心论文：**Open-TeleVision**（感知创新标杆）、**OPEN TEACH**（低成本通用）、**TWIST2**（whole-body 规模化）。

---

## 一、为什么需要 VR 人形 Teleop？

| 痛点（第三人称/静态相机 teleop） | VR + 主动头显的解法 |
|--------------------------------|---------------------|
| 机械臂/躯干遮挡视野 | **机器人头随人转头**，第一人称主动感知 |
| 精细任务看不清物体 | **立体视觉** + 注视点居中（foveal vision） |
| 同构 leader 无法控灵巧手 | **手部 keypoint → retarget** 到多指 hand |
| 操作员必须站在 lab | **跨洋远程** teleop（OTV：Boston ↔ San Diego） |

**与 ALOHA 对比**：ALOHA 用 **关节同构**（精度高、无 IK）；VR 用 **任务空间 + IK**（更通用、可远程、可 whole-body）。

---

## 二、Open-TeleVision（OTV）· 实现全流程 ⭐

> **论文精读（PDF 逐节解读）**：[Open-TeleVision-Paper-Walkthrough.md](./Open-TeleVision-Paper-Walkthrough.md)

**论文**：arXiv [2407.01512](https://arxiv.org/abs/2407.01512) · CoRL 2024  
**机构**：UC San Diego + MIT  
**本地 PDF**：`paper/Data-Collection/VR-Humanoid/Open-TeleVision.pdf`

### 2.1 系统架构（60 Hz 闭环）

```text
┌─────────────────────────────────────────────────────────────────┐
│  Apple Vision Pro（或其它 VR）                                   │
│  · 输出: hand / head / wrist pose ∈ SE(3) @ 60Hz                │
│  · 输入: 机器人双目 RGB 480×640 × 2                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ WebSocket / Vuer 服务器
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  TeleVision Server（基于 Vuer）                                  │
│  · human → robot motion retargeting                             │
│  · 臂: CLIK (Pinocchio)                                         │
│  · 手: dex-retargeting (SLSQP)                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ joint position targets
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  人形机器人 + 主动头显                                           │
│  · Unitree H1 + 6DoF Inspire 手 + 2DoF 自制 gimbal 颈           │
│  · Fourier GR-1 + jaw gripper + 3DoF 原厂颈                     │
│  · ZED Mini 立体相机                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 录制
                            ▼
              { stereo RGB, proprio, joint targets, language }
                            │
                            ▼
              ACT (DinoV2 + stereo) → 自主策略部署
```

### 2.2 硬件清单

| 组件 | H1 配置 | GR-1 配置 |
|------|---------|-----------|
| 机器人 | Unitree H1，7DoF×2 臂 + 6DoF×2 Inspire 手 | Fourier GR-1，7DoF×2 臂 + 1DoF jaw gripper |
| 主动颈 | **自制 2DoF gimbal**（yaw+pitch），3D 打印 + DYNAMIXEL XL330 | **原厂 3DoF 颈**（yaw+roll+pitch） |
| 相机 | **ZED Mini** 立体 RGB | 同左 |
| VR | **Apple Vision Pro**（论文主平台；系统声明 device-agnostic） | 同左 |
| 控制对象 | 颈 2DoF + 双臂 14DoF + 双手 12DoF = **28D action** | 颈 3DoF + 双臂 14DoF + 双夹爪 2DoF = **19D action** |

### 2.3 执行层：臂 + 手怎么映射

**臂控制（IK）**：

1. 把人腕 pose 变换到机器人坐标系  
2. **位置**：腕相对 **头** 的相对位置 → 机器人 EEF 相对机器人头（头动时 EEF 稳定）  
3. **朝向**：腕绝对朝向对齐人腕  
4. **CLIK**（Pinocchio 闭环 IK）+ SE(3) 平滑滤波  
5. 可操作度接近极限时，在 Jacobian 零空间加 joint offset，减少 IK 失败  

**手控制（retargeting）**：

- 库：**dex-retargeting**（vector optimizer + SLSQP）  
- **灵巧手（H1）**：7 条向量（腕→五指尖 + 拇指→食/中指）  
- **夹爪（GR-1）**：1 条向量（人拇食间距 → 夹爪开合）  
- 缩放因子 α=1.1（人手 vs Inspire 手尺寸差）  

### 2.4 感知层：为何「主动立体」是核心贡献

| 设计 | 作用 |
|------|------|
| **主动头显相机**（2–3 DoF 随人头动） | 操作员「看哪指哪」；策略也学 **主动转头** |
| **立体双目** 480×640/眼 | 深度/空间理解；User study：**Stereo >> Mono** |
| vs 静态广角 | 少 irrelevant pixel；训练 **2× 快**，推理 **83Hz vs 42Hz**（同 GPU） |

### 2.5 数采：录什么、怎么结束一条 demo

**每步 raw 信号**：

```text
observation:
  · stereo_left, stereo_right (RGB)
  · proprio: 当前全部 joint 读数（28D 或 19D）

action (监督标签):
  · 绝对 joint position targets（与 ACT 一致）
  · 含颈、臂、手/夹爪

language:
  · 任务级指令（episode 级）

metadata:
  · 任务类型、success、episode 边界
```

**Episode 边界**：操作员完成任务后做 **固定结束手势** → 系统停止录制（结束手势也留在数据里，策略可学「何时停」）。

**四类长时域任务**（论文 Fig.4）：

| 任务 | 特点 |
|------|------|
| Can Sorting | 10 罐连续分拣，颜色+位置泛化 |
| Can Insertion | 精细插入（≈5.6cm 罐 → 7.6cm 孔），拇+食二指捏 |
| Folding | 软体毛巾两次折叠 |
| Unloading | 视觉找 tube + 双手传递 |

### 2.6 策略训练（OTV 的 ACT 变体）

| 项 | OTV 相对原版 ACT |
|----|------------------|
| 视觉 backbone | **DinoV2-ViT**（替代 ResNet18） |
| 输入图像 | **2 路立体**（非 ALOHA 4 路固定相机） |
| Action | **绝对 joint**，chunk_size=**60**，60Hz 推理 |
| 损失 | L1 + KL（CVAE） |
| 训练 | AdamW lr=5e-5, batch=45, 25k iter, RTX 4090 |

**成功率（H1，DinoV2+Stereo）**：Can Sorting pick 92% / place 88%；Insertion 90%/87%；Folding/Unloading 多项 100%。

### 2.7 开源代码与运行

| 资源 | 链接 |
|------|------|
| **GitHub** | [OpenTeleVision/TeleVision](https://github.com/OpenTeleVision/TeleVision) |
| **官网+视频** | [robot-tv.github.io](https://robot-tv.github.io/) |
| 论文 HTML | [arXiv HTML](https://arxiv.org/html/2407.01512v2) |

**环境（README 摘要）**：

```bash
conda create -n tv python=3.8
conda activate tv
pip install -r requirements.txt
cd act/detr && pip install -e .
```

**Teleop 模式**：

```bash
# 本地
cd teleop && python teleop_hand.py   # 示例入口，见 repo

# 远程（跨洋）
# ngrok=True 开启，VR 经 internet 连 server
```

**训练**：

```bash
python imitate_episodes.py \
  --policy_class ACT --kl_weight 10 --chunk_size 60 \
  --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 \
  --num_epochs 50000 --lr 5e-5 --taskid 00 --exptid 01-sample-expt
```

### 2.8 视频资源（OTV）

[robot-tv.github.io](https://robot-tv.github.io/) 项目页含多段 **可点击播放** 的视频：

| 类别 | 内容 |
|------|------|
| **Remote Teleoperation** | 跨洋远程操作 |
| **Autonomous Skills** | 训好后自主执行（分拣/插入/折叠/卸载） |
| **Cross-Country** | Boston ↔ San Diego ~3000 miles |
| **Teleoperation** | 遥操作实录（含钻孔、耳塞包装、移液等） |

**第三方报道**（文字+配图）：[The Robot Report · OTV 介绍](https://www.therobotreport.com/mit-uc-san-diego-researchers-create-open-television-immersive-teleoperation/)

**B 站**：目前 OTV **无官方中文长讲解**；可先看官网 demo，再读本地 PDF + GitHub。

---

## 三、OPEN TEACH · 低成本通用 VR Teleop

**论文**：arXiv [2403.07870](https://arxiv.org/abs/2403.07870)  
**机构**：NYU / Meta 等  
**定位**：**$500 级 Meta Quest 3**，多 morphology，**全开源**

### 3.1 与 OTV 差异

| | Open-TeleVision | OPEN TEACH |
|---|-----------------|------------|
| VR 设备 | Apple Vision Pro | **Meta Quest 3** (~$500) |
| 感知 | **机器人主动立体**头显 | Quest passthrough + **机器人相机小窗** |
| 机器人 | 人形 H1/GR-1 | Franka/xArm/Jaco/**Allegro**/Hello Stretch |
| 频率 | 60 Hz | **最高 90 Hz** |
| 远程 | ✅ 跨洋 | ❌（强调 local MR） |
| 开源 | [TeleVision](https://github.com/OpenTeleVision/TeleVision) | [Open-Teach](https://github.com/aadhithya14/Open-Teach) |

### 3.2 实现要点

```text
Quest 3 onboard hand tracking @ 90Hz
  → Unity VR App（Open Teach APK）
  → 机器人 controller（NYU-robot-learning/OpenTeach-Controllers）
  → 支持仿真 + 真机 + 数据录制
  → BC / IRL 策略训练（10 任务 avg 87% success）
```

**支持平台**：Franka、xArm、Jaco、Allegro、Hello Stretch；**38 任务** demo。

### 3.3 视频与文档

| 资源 | 链接 |
|------|------|
| **官网（大量任务视频）** | [open-teach.github.io](https://open-teach.github.io/) |
| GitHub | [aadhithya14/Open-Teach](https://github.com/aadhithya14/Open-Teach) |
| Controllers | [NYU-robot-learning/OpenTeach-Controllers](https://github.com/NYU-robot-learning/OpenTeach-Controllers) |
| 文档 | repo 内 `docs/vr.md`, `docs/teleop_data_collect.md` |

官网视频分类：**Franka-Allegro / Kinova-Allegro / Bimanual / Allegro sim / LIBERO sim / Hello-Stretch / Policy Learning / User Study** —— 适合「看整个 workflow 长什么样」。

---

## 四、TWIST2 · 无 mocap 全身人形数采

**论文**：arXiv [2511.02832](https://arxiv.org/abs/2511.02832) · 2025  
**机构**：Amazon FAR + Berkeley 等  
**定位**：**whole-body**（含移动）+ **便携** + **fleet scale**

### 4.1 系统四件套

```text
1. PICO 4 Ultra + 2× PICO Motion Tracker  → 全身 pose @ VR
2. 自制 2DoF 机器人颈 (~$250)              → 主动 egocentric 视觉
3. Unitree G1                              → 全身体控
4. XRoboToolkit                            → 统一 PICO 应用（视觉+pose 流）
```

**规模**：**15 分钟 ~100 条** successful demo；128 次双臂 pick&place / 15 min。

### 4.2 与 OTV 差异

| | OTV | TWIST2 |
|---|-----|--------|
| 身体 DoF | mainly 颈+双臂+手 | **全身 locomotion + manipulation** |
| VR | Apple Vision Pro | **PICO 4 Ultra** |
| mocap | 不需要 | **不需要**（vs 传统 mocap 棚） |
| 策略 | ACT | **分层 visuomotor**（iDP3 改，单独 repo） |
| 数据集 | 论文 task | [twist-data.github.io](https://twist-data.github.io) 开源 |

### 4.3 视频与代码

| 资源 | 链接 |
|------|------|
| **YouTube Demo** | [youtu.be/lTtEvI0kUfo](https://youtu.be/lTtEvI0kUfo) |
| 项目页 | [yanjieze.com/TWIST2](https://yanjieze.com/projects/TWIST2/) |
| GitHub | [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2) |
| B 站复现 | [BV1UbSeBNETw](https://www.bilibili.com/video/BV1UbSeBNETw/)（社区复现，2025-12） |

**状态（2025-12）**：G1 + PICO 有线 sim/real 可复现；高层 policy repo 陆续放出。

---

## 五、端到端工作流（数采 → 训练）

以 **Open-TeleVision** 为主线（本仓库精读首选）：

```text
Phase 1 · 硬件搭建
  H1/GR-1 + ZED Mini + 主动颈 + Vision Pro
  网络: 本地 LAN 或 ngrok 远程

Phase 2 · Teleop 采集
  teleop_hand.py → 60Hz 闭环
  每 episode: stereo + joint targets + 任务指令
  结束手势 → 停录

Phase 3 · 前处理
  同步 (<10ms) → QA(IK fail/blur/idle) → HDF5/LeRobot

Phase 4 · 训练
  ACT + DinoV2 + stereo, chunk=60
  imitate_episodes.py

Phase 5 · 部署
  60Hz chunk 执行 + 闭环 re-infer
```

**Canonical IR 映射**：见 [总方案 §6.3](../../数采到VLA训练-数据管线整体方案.md)

---

## 六、多媒体学习路径（推荐顺序）

| 顺序 | 做什么 | 时间 |
|:--:|--------|:--:|
| 1 | 打开 [robot-tv.github.io](https://robot-tv.github.io/) 看 **Autonomous + Cross-Country** 视频 | 20 min |
| 2 | 读本地 `Open-TeleVision.pdf` **§2 System + §3 Experiments** | 1 h |
| 3 | 浏览 [open-teach.github.io](https://open-teach.github.io/) 对比「Quest3 低成本方案」 | 30 min |
| 4 | 看 TWIST2 [YouTube](https://youtu.be/lTtEvI0kUfo) 了解 whole-body scale | 10 min |
| 5 | Clone [TeleVision GitHub](https://github.com/OpenTeleVision/TeleVision)，读 `teleop/` + `act/` | 2 h |
| 6 | 对照 [ACT 原理](../ALOHA/ACT-Model-Working-Principles.md) 理解 OTV 改动点 | 1 h |

---

## 七、选型建议

| 场景 | 推荐 |
|------|------|
| 人形双臂+灵巧手+**主动立体**+远程 | **Open-TeleVision** |
| 桌面臂/ Allegro / Stretch，**预算 $500** | **OPEN TEACH** |
| Unitree G1 **全身移动操作**+ fleet 采数 | **TWIST2** |
| 低成本双臂桌面精细操作 | **ALOHA**（非 VR） |

---

## 八、关联

- **[Open-TeleVision 论文精读](./Open-TeleVision-Paper-Walkthrough.md)** ← PDF 对应笔记
- [数采总方案](../../数采到VLA训练-数据管线整体方案.md)
- [ACT 原理](../ALOHA/ACT-Model-Working-Principles.md)
- [资源索引 · OTV](../../resources/links/资源索引.md)
- [论文索引 §3 VR](../../../paper/论文索引.md)
