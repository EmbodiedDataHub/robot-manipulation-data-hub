---
title: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
authors: "Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song"
year: 2023
source: "arXiv:2303.04137"
tags:
  - Diffusion-Policy
  - imitation-learning
  - behavior-cloning
  - diffusion-models
  - visuomotor-policy
  - action-chunking
aliases:
  - Diffusion Policy 2023
paper_path: "paper/Algorithm/Diffusion-Policy/Diffusion Policy- Visuomotor Policy Learning via Action Diffusion.pdf"
code_path: "code/diffusion_policy"
project_website: "https://diffusion-policy.cs.columbia.edu"
---

## 一句话总结

Diffusion Policy 提出用 **扩散模型（DDPM）直接生成未来动作轨迹**，而不是逐步回归单步动作；配合 **receding horizon 执行** 与 **视觉条件注入**，在仿真与真机操作任务上同时获得 **更高成功率、更平滑轨迹、更强多模态拟合能力**。

---

## 论文基本信息

| 项 | 内容 |
|----|------|
| 标题 | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion |
| 作者 | Cheng Chi¹, Siyuan Feng², Yilun Du³, Zhenjia Xu¹, Eric Cousineau², Benjamin Burchfiel², Shuran Song¹ |
| 机构 | Columbia¹, Toyota Research Institute², MIT³ |
| 发表 | arXiv:2303.04137, 2023-03-07；RSS 2023 |
| 代码 | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) → 本仓库 [`code/diffusion_policy`](../../../code/diffusion_policy) |
| 数据 / 日志 | [项目页 Data](https://diffusion-policy.cs.columbia.edu/data/) |

---

## 1. 研究动机：逐步 BC 为什么不够

### 1.1 模仿学习的两个老问题

| 问题 | 表现 | 典型原因 |
|------|------|----------|
| **Compounding error** | 每步小偏差 → 进入 OOD 状态 → 误差滚雪球 | 单步 Markov 策略 horizon 太长 |
| **Multimodal demonstrations** | 同一场景人类/demo 有多种合理做法，MSE 学到「平均动作」 | 单峰回归无法表达多模态 |

精细操作、接触丰富任务里，这两个问题尤其致命：  
一条 demo 轨迹里可能有 **绕左 / 绕右** 两种合理路径，或者 **快推 / 慢推** 两种节奏；逐步 MLP/RNN 回归往往学到 **两边都不对** 的中间值。

### 1.2 已有缓解思路及其局限

| 方法 | 思路 | 局限 |
|------|------|------|
| **Action Chunking（ACT 等）** | 一次预测多步，缩短有效决策 horizon | 仍是单峰回归；多模态靠 CVAE 等额外机制 |
| **Energy-based / IBC** | 学习能量函数，采样低能量动作 | 推理慢、训练不稳定 |
| **BET（Behavior Transformer）** | 离散化动作 token，自回归生成 | 离散化损失精度；长 horizon 难 |

论文核心问题：

> **能否把图像生成里已验证的 diffusion 范式，原样迁移到「动作轨迹生成」，同时保持机器人控制的精度与速度？**

---

## 2. 核心思想：对「动作序列」做扩散，而不是对「单步动作」回归

### 2.1 从 Stable Diffusion 到 Action Diffusion

```
图像 Diffusion：  噪声图像  --去噪 T 步-->  清晰图像
Action Diffusion：噪声轨迹  --去噪 T 步-->  合理动作序列 a_{t:t+H}
```

- **扩散对象**：长度为 `horizon` 的动作轨迹 $\mathbf{A}_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]$
- **条件**：最近 `n_obs_steps` 帧观测（图像 + 低维状态等）
- **训练**：标准 DDPM——随机 timestep 加噪，网络预测噪声 ε
- **推理**：从 $\mathcal{N}(0,I)$ 采样噪声轨迹，迭代去噪得到动作序列

直觉：扩散模型天然是 **生成式、多模态** 的——同一观测可以采样出多条不同但合理的轨迹。

### 2.2 Receding Horizon Control（滚动时域）

一次预测 `horizon=16` 步，但环境 **只执行前 `n_action_steps=8` 步**，然后重新观测、重新采样：

```
t=0:  obs ──sample──> [a0..a15] ──execute──> a0..a7
t=8:  obs' ──sample──> [a8..a23] ──execute──> a8..a15
...
```

好处：

1. **缩短有效闭环 horizon**，减轻 compounding error
2. **在线重规划**，用新观测修正旧计划
3. 与 model predictive control (MPC) 精神一致，但 **模型是学出来的 diffusion policy**

---

## 3. 方法细节

### 3.1 观测条件化：两种注入方式

论文与代码支持两类条件机制：

| 模式 | 做法 | 适用 |
|------|------|------|
| **Global conditioning** | 观测编码为向量，通过 FiLM 注入 1D UNet 各层 | Push-T image、Robomimic image（默认） |
| **Inpainting conditioning** | 把 `[action \| obs_feature]` 拼成轨迹，mask 固定 obs 部分 | 部分 low-dim 设定 |

Push-T image 配置（`image_pusht_diffusion_policy_cnn.yaml`）使用 **global cond**：

- `n_obs_steps=2`：用最近 2 帧图像 + agent_pos
- 视觉编码：Robomimic 风格 CNN（Hybrid policy）或自定义 MultiImageObsEncoder

### 3.2 去噪网络：Conditional 1D UNet

- 输入 shape：`(B, horizon, action_dim)`，内部转为 `(B, channels, horizon)` 做 1D 卷积
- 条件：
  - **扩散 timestep** $k$ → sinusoidal embedding
  - **观测 global cond** → FiLM scale/bias 调制各 ResBlock
- 输出：预测噪声 ε（`prediction_type: epsilon`）或样本本身

论文还实验了 **Diffusion Transformer** 变体；代码里对应 `diffusion_transformer_*_policy.py`。

### 3.3 训练目标

标准 DDPM 损失，只在 **动作维度** 上计算（观测已通过条件注入）：

1. 取 demo 轨迹 $\mathbf{A}$，随机采样 timestep $k$
2. 加噪得 $\mathbf{A}_k$
3. 网络 $\epsilon_\theta(\mathbf{A}_k, k, \text{obs})$ 预测噪声
4. MSE$(\epsilon_\theta, \epsilon)$

与 BC 对比：BC 是 $\|\pi(o)-a\|^2$；Diffusion Policy 是 **生成整条轨迹分布**。

### 3.4 推理

1. 初始化 $\mathbf{A}_K \sim \mathcal{N}(0, I)$
2. for $k = K \ldots 1$：用 scheduler 一步去噪
3. 反归一化，取 `[To-1 : To-1+n_action_steps]` 作为执行动作

默认 `num_inference_steps=100`（与训练 timesteps 相同）；可截断加速。

---

## 4. 实验设置

### 4.1 任务与模态

| 类别 | 代表任务 | 观测 | 动作维 |
|------|----------|------|--------|
| **2D 接触操作** | Push-T | RGB 96×96 + agent_pos | 2D delta pos |
| **Robomimic** | Lift / Can / Square / Transport / Tool Hang | RGB + proprio | 7–20 |
| **真机** | Real Push-T | 双 RealSense + UR5 | 2D EE delta |
| **Long-horizon** | Kitchen (Franka) | low-dim state | 9 |

Push-T 是论文 **标志性 demo**：用 T 形块推入目标区域，接触丰富、轨迹多模态，最能体现 diffusion 优势。

### 4.2 Baseline

- **BC-RNN**：Robomimic 经典行为克隆
- **IBC-DFO**：隐式 BC + derivative-free optimization 采样
- **BET**：Behavior Transformer，离散动作 token

### 4.3 主要结论（Table I / II 量级）

- **Push-T image**：Diffusion Policy 达到 **~0.95+** mean score，显著高于 BC-RNN / IBC / BET
- **Robomimic 多任务**：在 Square、Tool Hang 等难任务上优势最大
- **真机 Push-T**：sim-to-real 可部署，成功率优于 baseline
- **Multimodal**：同一初始状态下可采样不同合理轨迹（论文 Fig. multimodal）

### 4.4 关键消融

| 因素 | 发现 |
|------|------|
| **horizon** | 太短欠规划，太长 compounding；16 是常用 sweet spot |
| **n_action_steps** | 与 horizon 配合；执行步数 < horizon 时 receding horizon 生效 |
| **obs steps** | 2 步历史观测通常足够提供速度/动态信息 |
| **EMA** | 评估用 EMA 权重更稳 |
| **网络规模** | UNet down_dims 增大提升难任务，但推理更慢 |

---

## 5. 系统视角：论文在整条技术线中的位置

```
BC / BC-RNN          单步或短序列回归，难处理多模态
    ↓
Diffusion Policy     生成式轨迹 + receding horizon  ← 本文
    ↓
ACT (ALOHA)          chunk + CVAE，另一条多模态路线
    ↓
UMI                  野外手持采集 + Diffusion Policy 策略接口
    ↓
RDT / π0             大模型 + diffusion / flow action head
```

| 对比维 | Diffusion Policy | ACT (ALOHA) |
|--------|------------------|-------------|
| 输出 | 长度 `horizon` 的 **连续轨迹** | 长度 `chunk_size` 的 **确定性解码 + CVAE** |
| 多模态 | 采样噪声 → 不同轨迹 | 训练时 latent z，推理 z=0 |
| 执行 | receding horizon，执行 `n_action_steps` | 逐步执行 chunk，每 100 步 re-query |
| 推理成本 | 多步去噪（~100 steps） | 单次前向 |
| 典型场景 | 接触丰富、轨迹分叉、UMI 人类 demo | 实验室 teleop、双臂精细操作 |

---

## 6. 读后应带走的三个设计原则

1. **生成对象要是「轨迹」不是「一步」** —— 把 MPC 里的 planning horizon 写进 policy 输出
2. **条件化要简单可复用** —— global FiLM cond 让同一 UNet 适配 image / low-dim
3. **训练用 demo，推理用 receding horizon** —— 兼顾 imitation 与闭环修正

---

## 7. 延伸阅读

| 文档 | 内容 |
|------|------|
| [模型工作原理](./Diffusion-Policy-Model-Working-Principles.md) | DDPM 公式 ↔ 代码 tensor |
| [代码导读](./DiffusionPolicy-Code-Walkthrough.md) | Push-T 训练/评估调用链 |
| [UMI 笔记](../UMI-Universal-Manipulation-Interface.md) | 为何 UMI 选 DP 而非 MLP |
| [ACT 原理](../ALOHA/ACT-Model-Working-Principles.md) | 另一条 chunk 路线 |
