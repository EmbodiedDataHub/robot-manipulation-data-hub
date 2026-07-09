---
title: "DiffusionPolicy 代码导读（逐层拆解）"
source: "code/diffusion_policy/"
tags:
  - Diffusion-Policy
  - code-walkthrough
  - training
  - inference
  - hydra
related:
  - "Diffusion-Policy-Model-Working-Principles.md"
  - "Diffusion-Policy-Paper-Walkthrough.md"
code_path: "code/diffusion_policy"
---

# DiffusionPolicy 代码导读

> **定位**：以 **Push-T image + DiffusionUnetHybridImagePolicy** 为主线，自顶向下 6 层拆解官方代码。  
> **读法**：先读 [模型工作原理](./Diffusion-Policy-Model-Working-Principles.md) 建立 tensor 概念，再按层追代码。  
> **姊妹篇**：[论文精读](./Diffusion-Policy-Paper-Walkthrough.md)

---

## 0. 30 秒总结

```text
入口 train.py
  → Hydra 加载 config（如 image_pusht_diffusion_policy_cnn.yaml）
  → TrainDiffusionUnetHybridWorkspace.run()
      → PushTImageDataset：zarr 滑窗 (horizon=16)
      → DiffusionUnetHybridImagePolicy.compute_loss()：DDPM ε-loss
      → 每 epoch：PushTImageRunner.run(policy) 评估成功率
  → checkpoint 存 cfg + state_dict（dill 序列化）

推理 eval.py
  → load checkpoint → policy.predict_action(obs)
  → MultiStepWrapper 环境逐步执行 n_action_steps
```

**Push-T 典型超参**（`image_pusht_diffusion_policy_cnn.yaml`）：

| 键 | 值 |
|----|-----|
| `horizon` | 16 |
| `n_obs_steps` | 2 |
| `n_action_steps` | 8 |
| `num_inference_steps` | 100 |
| `obs_as_global_cond` | true |
| `prediction_type` | epsilon |

---

## 1. 第 0 层：仓库结构与入口

### 1.1 目录地图

```
code/diffusion_policy/
├── train.py                    # ★ 统一训练入口（Hydra）
├── eval.py                     # ★ 加载 ckpt 评估
├── demo_pusht.py               # 鼠标 teleop 采 Push-T demo
├── image_pusht_diffusion_policy_cnn.yaml   # Push-T 官方 config
└── diffusion_policy/
    ├── workspace/              # 训练 orchestration
    ├── policy/                 # ★ Diffusion*Policy
    ├── model/diffusion/        # ConditionalUnet1D, EMA
    ├── dataset/                # PushT*Dataset, Robomimic*
    ├── env_runner/             # PushTImageRunner 等
    └── env/pusht/              # Push-T 仿真环境
```

### 1.2 训练入口 `train.py`

```python
# train.py
@hydra.main(config_path='diffusion_policy/config')
def main(cfg):
    cls = hydra.utils.get_class(cfg._target_)   # 如 TrainDiffusionUnetHybridWorkspace
    workspace = cls(cfg)
    workspace.run()
```

**典型命令**（README）：

```bash
conda activate robodiff
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Hydra 根据 yaml 里 `_target_` 字段实例化 workspace、policy、dataset、env_runner。

### 1.3 评估入口 `eval.py`

```python
payload = torch.load(checkpoint, pickle_module=dill)
workspace.load_payload(payload)
policy = workspace.ema_model if cfg.training.use_ema else workspace.model
env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
runner_log = env_runner.run(policy)
```

---

## 2. 第 1 层：Workspace 训练循环

**文件**：[`workspace/train_diffusion_unet_image_workspace.py`](../../../code/diffusion_policy/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py)  
（Hybrid 版逻辑相同，类名 `TrainDiffusionUnetHybridWorkspace`）

### 2.1 初始化

```python
self.model = hydra.utils.instantiate(cfg.policy)          # DiffusionUnet*Policy
self.ema_model = copy.deepcopy(self.model) if use_ema else None
self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
```

### 2.2 `run()` 主流程

```
1. dataset = hydra.utils.instantiate(cfg.task.dataset)
2. normalizer = dataset.get_normalizer()
3. model.set_normalizer(normalizer)
4. env_runner = hydra.utils.instantiate(cfg.task.env_runner)
5. for epoch:
     a. batch → model.compute_loss(batch) → backward
     b. ema.step(model)
     c. 每 rollout_every：env_runner.run(ema_model) → test/mean_score
     d. 每 val_every：val_loss
     e. 每 checkpoint_every：save top-k ckpt
```

### 2.3 一个 training step

```python
batch = dict_apply(batch, lambda x: x.to(device))
raw_loss = self.model.compute_loss(batch)
loss = raw_loss / gradient_accumulate_every
loss.backward()
optimizer.step(); lr_scheduler.step()
```

**batch 形状**（Push-T image）：

```python
batch['obs']['image']     # (B, T, 3, 96, 96)
batch['obs']['agent_pos'] # (B, T, 2)
batch['action']           # (B, T, 2)   T = horizon = 16
```

---

## 3. 第 2 层：Policy — 训练与推理

**主文件**：

- [`policy/diffusion_unet_image_policy.py`](../../../code/diffusion_policy/diffusion_policy/policy/diffusion_unet_image_policy.py) — 自定义 CNN encoder
- [`policy/diffusion_unet_hybrid_image_policy.py`](../../../code/diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py) — **Push-T 默认**，Robomimic encoder

两者 `compute_loss` / `predict_action` / `conditional_sample` **逻辑相同**，差别在 obs_encoder 构建。

### 3.1 `compute_loss`（训练）

```python
def compute_loss(self, batch):
    nobs = self.normalizer.normalize(batch['obs'])
    nactions = self.normalizer['action'].normalize(batch['action'])
    # global cond 路径
    this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
    nobs_features = self.obs_encoder(this_nobs)
    global_cond = nobs_features.reshape(batch_size, -1)

    trajectory = nactions                              # (B, H, Da)
    noise = torch.randn(trajectory.shape, ...)
    timesteps = torch.randint(0, num_train_timesteps, (bsz,))
    noisy_trajectory = noise_scheduler.add_noise(trajectory, noise, timesteps)

    pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
    loss = F.mse_loss(pred, noise)   # epsilon target
    return loss
```

### 3.2 `predict_action`（推理）

```python
def predict_action(self, obs_dict):
    nobs = self.normalizer.normalize(obs_dict)
    # encode 最近 n_obs_steps 帧
    global_cond = obs_encoder(...).reshape(B, -1)
    cond_data = zeros(B, horizon, action_dim)
    nsample = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond)
    action_pred = normalizer['action'].unnormalize(nsample)
    action = action_pred[:, To-1 : To-1+n_action_steps]
    return {'action': action, 'action_pred': action_pred}
```

### 3.3 `conditional_sample`（DDPM 循环）

```python
trajectory = torch.randn(size=condition_data.shape, ...)
scheduler.set_timesteps(num_inference_steps)
for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]
    model_output = model(trajectory, t, global_cond=global_cond)
    trajectory = scheduler.step(model_output, t, trajectory).prev_sample
return trajectory
```

**调试建议**：在 `predict_action` 出口打印 `action.shape`，应为 `(B, n_action_steps, Da)`。

---

## 4. 第 3 层：ConditionalUnet1D + Obs Encoder

### 4.1 ConditionalUnet1D

**文件**：[`model/diffusion/conditional_unet1d.py`](../../../code/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py)

```python
# forward(x, timesteps, local_cond=None, global_cond=None)
# x: (B, H, input_dim) — global cond 模式下 input_dim = action_dim
timestep_embed = self.diffusion_step_encoder(timesteps)
if global_cond is not None:
    global_feature = torch.cat([timestep_embed, global_cond], dim=-1)
# UNet down → mid → up，每层 ConditionalResidualBlock1D(x, global_feature)
```

FiLM 调制在 `ConditionalResidualBlock1D`：

```python
scale, bias = cond_encoder(global_feature).chunk(2, dim=1)
out = scale * out + bias
```

### 4.2 Hybrid Obs Encoder（Push-T）

`DiffusionUnetHybridImagePolicy` 用 **Robomimic** 的 `ObservationEncoder`：

- 读取 `shape_meta.obs`：`image (3,96,96)` + `agent_pos (2,)`
- `CropRandomizer` 训练时随机 crop，eval 时 fixed crop
- 输出 flatten 后拼成 `global_cond`

自定义版 `DiffusionUnetImagePolicy` 则用 `MultiImageObsEncoder`，逻辑等价。

---

## 5. 第 4 层：Dataset 与 Normalizer

**文件**：[`dataset/pusht_image_dataset.py`](../../../code/diffusion_policy/diffusion_policy/dataset/pusht_image_dataset.py)

### 5.1 数据来源

```python
ReplayBuffer.copy_from_path(zarr_path, keys=['img', 'state', 'action'])
SequenceSampler(replay_buffer, sequence_length=horizon, ...)
```

官方数据：`data/pusht/pusht_cchi_v7_replay.zarr`（wget 下载 pusht.zip）

### 5.2 `__getitem__` 输出

```python
{
  'obs': {
    'image': (T, 3, 96, 96),   # float, 已 /255
    'agent_pos': (T, 2),        # state 前两维 = agent xy
  },
  'action': (T, 2)              # 2D delta position
}
```

### 5.3 Normalizer

```python
normalizer.fit({'action': ..., 'agent_pos': state[...,:2]}, mode='limits')
normalizer['image'] = get_image_range_normalizer()
```

Policy 在 workspace 里 `set_normalizer(normalizer)`，保证 train/eval 尺度一致。

---

## 6. 第 5 层：Env Runner — Rollout 闭环

**文件**：[`env_runner/pusht_image_runner.py`](../../../code/diffusion_policy/diffusion_policy/env_runner/pusht_image_runner.py)

### 6.1 环境包装链

```
PushTImageEnv
  → VideoRecordingWrapper（存 mp4）
  → MultiStepWrapper(n_obs_steps, n_action_steps, max_episode_steps)
  → AsyncVectorEnv（并行多个 seed）
```

### 6.2 Rollout 核心循环

```python
obs = env.reset()
policy.reset()
while not done:
    obs_dict = dict_apply(obs, torch.from_numpy)
    action_dict = policy.predict_action(obs_dict)
    action = action_dict['action'].cpu().numpy()   # (B, n_action_steps, Da)
    obs, reward, done, info = env.step(action)
```

### 6.3 指标

```python
max_reward = np.max(all_rewards[i])   # Push-T reward 与 T 块覆盖目标区相关
log_data['test/mean_score'] = np.mean(max_rewards)
```

论文 Table 里的 success rate 对应 `test/mean_score`（wandb 同名）。

---

## 7. 第 6 层：数据采集与 Config

### 7.1 采 demo — `demo_pusht.py`

```bash
python demo_pusht.py -o data/pusht_demo.zarr
```

鼠标 teleop → 写入 zarr replay buffer（`img, state, action`）。官方训练用的是作者预采的大规模 zarr。

### 7.2 Config 关键字段解读

```yaml
# image_pusht_diffusion_policy_cnn.yaml（节选）
_target_: ...TrainDiffusionUnetHybridWorkspace
horizon: 16
n_obs_steps: 2
n_action_steps: 8
obs_as_global_cond: true
policy:
  _target_: ...DiffusionUnetHybridImagePolicy
  noise_scheduler:
    num_train_timesteps: 100
    prediction_type: epsilon
task:
  dataset:
    _target_: ...PushTImageDataset
    zarr_path: data/pusht/pusht_cchi_v7_replay.zarr
  env_runner:
    _target_: ...PushTImageRunner
training:
  use_ema: true
  rollout_every: 50
```

改实验优先动：`horizon`、`n_action_steps`、`down_dims`、`num_inference_steps`。

---

## 8. 其他 Policy 变体（选读）

| 文件 | 场景 |
|------|------|
| `diffusion_unet_lowdim_policy.py` | 纯 state，无图像 |
| `diffusion_transformer_*_policy.py` | Transformer 去噪网络 |
| `robomimic_*_policy.py` | Baseline BC |
| `ibc_dfo_*_policy.py` | IBC baseline |

Workspace 与 `train.py` 入口 **完全共用**，仅 yaml 里 `_target_` 不同。

---

## 9. 与 ALOHA `code/act` 的调用风格对比

| 维度 | Diffusion Policy | ALOHA ACT |
|------|------------------|-----------|
| 配置 | Hydra yaml | argparse |
| 训练脚本 | `train.py` + workspace | `imitate_episodes.py` |
| Policy API | `compute_loss(batch)` / `predict_action(obs_dict)` | `policy(qpos, image, actions, pad)` |
| 评估 | `env_runner.run(policy)` | `eval_bc()` 手写循环 |
| 数据 | zarr replay buffer | HDF5 episode |

---

## 10. 动手 Checklist

- [ ] `mamba env create -f conda_environment.yaml`（Linux + GPU）
- [ ] 下载 `pusht.zip` 到 `data/pusht/`
- [ ] `python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42`
- [ ] wandb 查看 `test/mean_score` 曲线
- [ ] `python eval.py -c <ckpt> -o /tmp/pusht_eval`
- [ ] 改 `num_inference_steps=10` 看速度/成功率 trade-off

---

## 11. 源码索引

| 你想… | 打开 |
|--------|------|
| 改 loss | `diffusion_unet_image_policy.py` → `compute_loss` |
| 改采样步数 | yaml `num_inference_steps` 或 policy kwargs |
| 改网络宽度 | yaml `down_dims` |
| 改数据窗口 | yaml `horizon` + dataset `SequenceSampler` |
| 改评估环境 | yaml `task.env_runner` |
| 存 ckpt 格式 | `workspace/base_workspace.py` → `save_checkpoint` |

算法原理详见 [Diffusion-Policy-Model-Working-Principles.md](./Diffusion-Policy-Model-Working-Principles.md)。
