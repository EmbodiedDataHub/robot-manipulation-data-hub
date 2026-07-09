---
title: "Diffusion Policy 模型工作原理"
authors: "Cheng Chi, Siyuan Feng, et al. (Columbia / TRI)"
year: 2023
source: "code/diffusion_policy (Diffusion Policy: Visuomotor Policy Learning via Action Diffusion)"
tags:
  - Diffusion-Policy
  - DDPM
  - imitation-learning
  - action-chunking
  - receding-horizon
aliases:
  - Action Diffusion
code_path: "code/diffusion_policy"
related:
  - "Diffusion-Policy-Paper-Walkthrough.md"
  - "DiffusionPolicy-Code-Walkthrough.md"
---

## 一句话总结

Diffusion Policy 把 **长度为 horizon 的动作轨迹** 当作 DDPM 的生成目标：训练时随机 timestep 加噪并预测 ε；推理时从纯噪声出发迭代去噪，再按 **receding horizon** 只执行前 `n_action_steps` 步。

---

## 1. 总体数据流

```
训练 demo 序列
  obs[t-To+1 : t]  +  action[t : t+H-1]
           │
           ▼
    ┌──────────────────┐
    │ Obs Encoder (CNN)│──> global_cond (B, To·Do)
    └──────────────────┘
           │
    action trajectory (B, H, Da) ──加噪──> noisy_traj
           │
           ▼
    ┌──────────────────┐
    │ ConditionalUnet1D│ + timestep emb + FiLM(global_cond)
    └──────────────────┘
           │
           ▼
      MSE(pred_ε, true_ε)

推理
  obs ──> global_cond
  rand noise (B, H, Da) ──K步去噪──> action_pred ──切片──> execute (B, n_action_steps, Da)
```

**符号对照**（与代码一致）：

| 符号 | 代码变量 | Push-T image 典型值 |
|------|----------|---------------------|
| H | `horizon` | 16 |
| To | `n_obs_steps` | 2 |
| 执行步数 | `n_action_steps` | 8 |
| Da | `action_dim` | 2 |
| Do | `obs_feature_dim` | CNN 输出维 |

---

## 2. 三个时间尺度：horizon / obs / action

这是读 Diffusion Policy **最容易混淆** 的地方。

### 2.1 horizon（预测多长）

模型一次生成 **H 步动作**，shape `(B, H, Da)`。

```python
# diffusion_unet_image_policy.py
T = self.horizon          # 16
Da = self.action_dim      # 2
cond_data = torch.zeros(size=(B, T, Da), ...)
```

### 2.2 n_obs_steps（看多远的历史）

条件化用 **最近 To 帧观测**（图像 stack + agent_pos 等）：

```python
this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
nobs_features = self.obs_encoder(this_nobs)
global_cond = nobs_features.reshape(B, -1)   # (B, To * Do)
```

Dataset 里每条样本就是长度 `horizon` 的序列；`SequenceSampler` 从 replay buffer 滑窗截取。

### 2.3 n_action_steps（实际执行几步）

推理后 **不执行整条 horizon**，只取：

```python
start = To - 1          # 1，当 To=2
end = start + self.n_action_steps   # 9
action = action_pred[:, start:end]  # (B, 8, 2)
```

**含义**：用 `t-1` 和 `t` 两帧观测，预测从 `t` 开始的未来动作；执行 8 步后重新采样。

### 2.4 与 ACT 的「chunk」对照

| | Diffusion Policy | ACT |
|--|------------------|-----|
| 一次输出 | H 步动作轨迹 | chunk_size 步（如 100） |
| 重新规划 | 每 `n_action_steps` 步 | 每 `chunk_size` 步（或 temporal agg 每步） |
| 多模态 | 采样噪声不同 → 轨迹不同 | CVAE latent z（推理 z=0） |
| 核心网络 | 1D Conditional UNet + DDPM | ResNet + Transformer Decoder |

---

## 3. DDPM 在动作轨迹上怎么工作

### 3.1 前向扩散（训练时加噪）

```python
noise = torch.randn(trajectory.shape, device=trajectory.device)
timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=...)
noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
```

- `trajectory` = 归一化后的 demo 动作 `(B, H, Da)`
- 每个 batch 样本随机一个 `k ∈ [0, K-1]`
- scheduler 来自 HuggingFace `diffusers.DDPMScheduler`（`prediction_type: epsilon`）

### 3.2 反向去噪（推理时采样）

```python
# conditional_sample()
trajectory = torch.randn(size=condition_data.shape, ...)
scheduler.set_timesteps(self.num_inference_steps)
for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]
    model_output = model(trajectory, t, global_cond=global_cond)
    trajectory = scheduler.step(model_output, t, trajectory).prev_sample
```

- 从 **纯高斯噪声** 出发
- 每步：网络预测 ε → scheduler 更新 `x_{k-1}`
- `condition_mask` 在 DP 的 global-cond 模式下全 False（动作维全部参与去噪）

### 3.3 损失函数

```python
pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
target = noise   # epsilon prediction
loss = F.mse_loss(pred, target, reduction='none')
loss = loss * loss_mask.type(loss.dtype)
loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
```

**没有** 对单步 action 做 BC MSE，而是对 **整条轨迹上的噪声预测** 做 MSE。

---

## 4. 观测条件化：Global Cond vs Inpainting

### 4.1 Global conditioning（默认，推荐先理解）

`obs_as_global_cond=True`：

- UNet 输入 **只有动作** `(B, H, Da)`
- 观测 → CNN → flatten → `global_cond (B, To·Do)`
- 在 `ConditionalResidualBlock1D` 里用 **FiLM** 调制：

```python
# conditional_unet1d.py — FiLM scale & bias
embed = self.cond_encoder(cond)   # cond 含 timestep + global_cond
scale = embed[:,0,...]; bias = embed[:,1,...]
out = scale * out + bias
```

### 4.2 Inpainting conditioning

`obs_as_global_cond=False`：

- 轨迹 tensor = `[action | obs_feature]`，维度 `Da + Do`
- mask 固定前 To 步的 obs 部分，只对 action 部分去噪
- 类似「已知观测序列、补全动作序列」

Push-T image 实验用 **global cond**；读代码时先掌握这条路径即可。

---

## 5. ConditionalUnet1D 结构

```
输入 x: (B, H, Da) → rearrange → (B, Da, H)

Encoder path (down):
  ResBlock + Downsample × len(down_dims)
  每级 down_dims: [512, 1024, 2048] (Push-T hybrid)

Bottleneck:
  ResBlock × 2

Decoder path (up):
  Upsample + ResBlock（skip connection）

条件注入:
  diffusion_step_embed(k) ⊕ global_cond → 每个 ResBlock 的 FiLM

输出: (B, Da, H) → rearrange → (B, H, Da) 预测 ε
```

**为何用 1D UNet 而不是 Transformer？**

- 动作轨迹是 **有序 1D 序列**，局部平滑性重要
- 1D conv 参数效率高，推理比 Transformer 版快
- 论文两者都试了；Push-T 上 CNN UNet 已足够强

---

## 6. 归一化（Normalizer）

```python
# PushTImageDataset.get_normalizer()
normalizer.fit({'action': ..., 'agent_pos': state[..., :2]}, mode='limits')
normalizer['image'] = get_image_range_normalizer()  # [0,1] → [-1,1]
```

- **action / low_dim**：按数据集 min-max 或 limits 归一化到 ~[-1, 1]
- **image**：除以 255 再映射到 [-1, 1]
- Policy 在 `set_normalizer()` 后，`compute_loss` / `predict_action` 内自动 normalize / unnormalize

---

## 7. Receding Horizon 在 rollout 里怎么体现

`PushTImageRunner` + `MultiStepWrapper`：

```python
# env 包装：一次 step 接收 (n_action_steps, Da) 的动作块
MultiStepWrapper(..., n_obs_steps=8, n_action_steps=8, max_episode_steps=200)

# rollout 循环
action_dict = policy.predict_action(obs_dict)  # action: (B, 8, 2)
obs, reward, done, info = env.step(action)
```

- `MultiStepWrapper` 内部维护 obs 历史栈，凑够 `n_obs_steps` 帧再喂给 policy
- 每调一次 `predict_action` = **一次 diffusion 采样** = 一次 receding horizon 规划
- 200 max steps / 8 action steps ≈ 25 次 re-plan

---

## 8. EMA（Exponential Moving Average）

训练 workspace 维护 `ema_model`：

```python
if cfg.training.use_ema:
    ema.step(self.model)
# 评估 rollout 时用 ema_model
policy = self.ema_model
runner_log = env_runner.run(policy)
```

扩散模型训练噪声大，EMA 权重在推理时更稳定——这是 diffusion 训 policy 的常见技巧。

---

## 9. 多模态从哪来？

同一条观测 `obs`，不同随机种子 → 不同初始噪声 → 不同去噪轨迹：

```
obs 固定 + seed=1  → 轨迹 A（从左侧绕 T 块）
obs 固定 + seed=2  → 轨迹 B（从右侧推）
```

逐步 MSE 会学成 $\frac{A+B}{2}$，往往 **撞墙**；  
Diffusion 保留 **两个模态** 作为分布的两个采样点。

UMI 选 Diffusion Policy 的核心原因：人类野外 demo **节奏、路径、停顿** 高度多模态。

---

## 10. 与 BC-RNN / IBC 的机制差异（复习）

| 方法 | 输出 | 训练 | 推理 |
|------|------|------|------|
| BC-RNN | 单步 $a_t$ | $\|π-o_t\|^2$ | 一次 forward |
| IBC | 单步 $a_t$ | 能量 $E(o,a)$ | 迭代采样优化 |
| **Diffusion Policy** | 轨迹 $\mathbf{A}_{t:t+H}$ | DDPM ε-pred | K 步去噪 |
| ACT | chunk $\mathbf{A}_{t:t+C}$ | L1 + KL | 一次 forward |

---

## 11. 源码速查表

| 问题 | 文件 | 函数 |
|------|------|------|
| 训练 loss | `policy/diffusion_unet_image_policy.py` | `compute_loss` |
| 推理采样 | 同上 | `predict_action` → `conditional_sample` |
| UNet | `model/diffusion/conditional_unet1d.py` | `ConditionalUnet1D.forward` |
| 训练循环 | `workspace/train_diffusion_unet_image_workspace.py` | `TrainDiffusionUnetImageWorkspace.run` |
| 数据格式 | `dataset/pusht_image_dataset.py` | `__getitem__` |
| Rollout | `env_runner/pusht_image_runner.py` | `run` |

详细调用链见 [DiffusionPolicy 代码导读](./DiffusionPolicy-Code-Walkthrough.md)。
