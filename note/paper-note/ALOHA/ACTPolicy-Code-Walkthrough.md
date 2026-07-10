---
title: "ACTPolicy 代码导读（逐层拆解）"
source: "code/act/"
tags:
  - ACT
  - ALOHA
  - CVAE
  - code-walkthrough
  - training
  - inference
related:
  - "CNNMLPPolicy-Code-Walkthrough.md"
  - "ACT-Model-Working-Principles.md"
  - "ALOHA-Learning-Fine-Grained-Bimanual-Manipulation.md"
code_path: "code/act"
---

# ACTPolicy 代码导读

> **定位**：ALOHA 论文主方法 —— **CVAE Encoder（仅训练）+ 共享 ResNet + Transformer Decoder**，一次输出 **chunk_size（默认 100）步** 关节目标。  
> **读法**：自顶向下 6 层；公共框架（`imitate_episodes` / `utils`）建议先读 [CNNMLPPolicy 导读](./CNNMLPPolicy-Code-Walkthrough.md) §1、§6。  
> **姊妹篇**：[CNNMLPPolicy 代码导读](./CNNMLPPolicy-Code-Walkthrough.md) · 原理深读 [ACT 模型工作原理](./ACT-Model-Working-Principles.md)

---

## 0. 30 秒总结

```text
训练：
  (image, qpos) + 未来 100 步 demo actions
    → CVAE Encoder：actions → μ, logσ² → z (32维)
    → 共享 ResNet18 提视觉特征
    → Transformer Decoder：100 queries × cross-attn
    → â (B, 100, 14)
    → loss = L1(â, a, mask pad) + β·KL

推理：
  (image, qpos) only · Encoder 关闭 · z = 0
    → 同上 Decoder → chunk (1, 100, 14)
    → eval_bc：每 100 步 query 一次，逐步执行 chunk[i]
    → 可选 temporal_agg：重叠预测指数加权平均
```

### 0.1 模型结构图

![CNNMLP vs ACT 对照](./assets/cnnmlp-vs-act-comparison.svg)

![ACT DETRVAE 训练/推理结构](./assets/act-detrvae-architecture.svg)

![CNNMLPPolicy 前向结构（baseline 对照）](./assets/cnnmlp-architecture.svg)

---

## 1. 第 0 层：谁调用 ACTPolicy？

**文件**：[`code/act/imitate_episodes.py`](../../../code/act/imitate_episodes.py)

### 1.1 创建 Policy

```python
# L53-68：ACT 专用超参
if policy_class == 'ACT':
    policy_config = {
        'lr': args['lr'],
        'num_queries': args['chunk_size'],      # ★ 默认 100 = chunk 长度
        'kl_weight': args['kl_weight'],        # ★ 默认 10
        'hidden_dim': args['hidden_dim'],      # 512
        'dim_feedforward': args['dim_feedforward'],  # 3200
        'lr_backbone': lr_backbone,             # 1e-5
        'backbone': 'resnet18',
        'enc_layers': 4,                        # CVAE Encoder
        'dec_layers': 7,                        # Transformer Decoder
        'nheads': 8,
        'camera_names': camera_names,
    }

# L121-123
def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
```

**典型训练命令**（`code/act/README.md`）：

```bash
python imitate_episodes.py \
  --policy_class ACT --kl_weight 10 --chunk_size 100 \
  --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  ...
```

### 1.2 训练循环

与 CNNMLP **完全相同**的入口：

```python
# L316-319
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    return policy(qpos_data, image_data, action_data, is_pad)
    # actions 非 None → ACTPolicy 走训练分支
```

### 1.3 评估循环（chunk + temporal_agg）

```python
# L191-194：query 频率
query_frequency = policy_config['num_queries']   # 默认 100
if temporal_agg:
    query_frequency = 1                        # TE 模式下每步都 query

# L247-261：ACT 专用
if config['policy_class'] == "ACT":
    if t % query_frequency == 0:
        all_actions = policy(qpos, curr_image)   # (1, 100, 14)
    if temporal_agg:
        # 把每次 query 的 chunk 写入大矩阵，对「预测同一时刻 t」的多条轨迹加权
        all_time_actions[[t], t:t+num_queries] = all_actions
        ...
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
    else:
        raw_action = all_actions[:, t % query_frequency]  # 从 chunk 里取第 i 步
```

| 模式 | query 频率 | 执行哪一步 |
|------|-----------|-----------|
| 默认（无 TE） | 每 **100** 步 | `all_actions[:, t % 100]` |
| `--temporal_agg` | **每步** query | 对历史重叠预测做指数加权平均 |

对比 CNNMLP：每步 `policy(qpos, image)` → `(1, 14)`，无 chunk。

---

## 2. 第 1 层：`ACTPolicy` — 训练/推理分叉

**文件**：[`code/act/policy.py`](../../../code/act/policy.py) L9-64

### 2.1 类结构

```python
class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model          # DETRVAE
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
```

对外 **唯一入口**：`__call__(qpos, image, actions=None, is_pad=None)`

### 2.2 训练分支（`actions is not None`）

```python
# L36-54
image = normalize(image)   # ImageNet

actions = actions[:, :self.model.num_queries]   # (B, 100, 14)
is_pad  = is_pad[:, :self.model.num_queries]    # (B, 100)

a_hat, is_pad_hat, (mu, logvar) = self.model(
    qpos, image, env_state, actions, is_pad)

total_kld, _, _ = kl_divergence(mu, logvar)

all_l1 = F.l1_loss(actions, a_hat, reduction='none')          # (B,100,14)
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()                  # mask padding

loss = l1 + total_kld[0] * self.kl_weight
return {'l1': l1, 'kl': total_kld[0], 'loss': loss}
```

要点：

| 项 | 说明 |
|----|------|
| `actions[:, :num_queries]` | 监督 **整段 future 100 步**，不是 CNNMLP 的 `[:, 0]` |
| `is_pad` | **真正参与 loss** —— episode 末尾 pad 步不计入 L1 |
| `is_pad_hat` | 网络也预测 pad，但 **policy 层未对其算 loss**（可扩展） |
| L1 vs MSE | 对 outlier 更鲁棒 |
| KL | 把 `q(z\|actions)` 拉向 `N(0,I)`；`kl_weight` 默认 10 |

### 2.3 推理分支（`actions is None`）

```python
# L55-58
a_hat, _, (_, _) = self.model(qpos, image, env_state)
return a_hat    # (B, num_queries, 14) = (B, 100, 14)
```

- **不传** `actions` / `is_pad`
- `DETRVAE.forward` 内 `is_training=False` → **跳过 CVAE Encoder**，`z = zeros(32)`
- 返回整段 chunk，由 `eval_bc` 逐步消费

### 2.4 训练 vs 推理（Policy 层）

| | 训练 | 推理 |
|---|------|------|
| 调用 | `policy(qpos, img, actions, is_pad)` | `policy(qpos, img)` |
| CVAE Encoder | **运行**（actions → z） | **关闭** |
| latent z | `reparametrize(μ, logσ²)` | **固定 0** |
| model 输入 | qpos + image + actions | qpos + image |
| 输出 | `(B, 100, 14)` | `(B, 100, 14)` |
| 返回 | `{l1, kl, loss}` | `a_hat` chunk |

### 2.5 观测侧仍是一阶马尔可夫

策略 **不读历史帧**；每步 query 时只用当前 `(image_t, qpos_t)`。  
与 CNNMLP 相同：观测条件是一阶马尔可夫；差别在于 **输出是 100 步 chunk**，在动作层面做了短 horizon 规划。

---

## 3. 第 2 层：模型构建与优化器

**文件**：[`code/act/detr/main.py`](../../../code/act/detr/main.py) L70-90

```python
def build_ACT_model_and_optimizer(args_override):
    model = build_ACT_model(args)    # → detr/models/__init__.py → build()
    model.cuda()
    param_dicts = [
        {"params": [非 backbone 参数], "lr": args.lr},           # 1e-4
        {"params": [backbone 参数],     "lr": args.lr_backbone}, # 1e-5
    ]
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)
    return model, optimizer
```

**文件**：[`code/act/detr/models/detr_vae.py`](../../../code/act/detr/models/detr_vae.py) L229-255

```python
def build(args):
    backbone = build_backbone(args)
    backbones = [backbone]              # ★ 只建 1 个，所有相机共享

    transformer = build_transformer(args)  # 7 dec layers
    encoder = build_encoder(args)          # 4 enc layers（CVAE 用）

    model = DETRVAE(
        backbones, transformer, encoder,
        state_dim=14,
        num_queries=args.num_queries,      # = chunk_size
        camera_names=args.camera_names,
    )
    return model
```

默认规模（README）：hidden_dim=512，参数量约 **~80M** 量级（随配置变化）。

---

## 4. 第 3 层：`DETRVAE` 网络结构

**文件**：[`code/act/detr/models/detr_vae.py`](../../../code/act/detr/models/detr_vae.py) L34-139

### 4.1 模块清单

```text
DETRVAE
├── encoder (CVAE)                    # 仅训练时
│     ├── cls_embed, encoder_action_proj, encoder_joint_proj
│     ├── TransformerEncoder (4 layers)
│     └── latent_proj → μ, logvar (32-d)
├── backbones[0]                      # 共享 ResNet18 + sine pos enc
├── input_proj                        # Conv1×1: 512 → hidden_dim
├── input_proj_robot_state            # Linear(14 → hidden_dim)
├── latent_out_proj                   # Linear(32 → hidden_dim)
├── query_embed                       # Embedding(100, hidden_dim)
├── transformer (Decoder 侧)          # 7 layers, cross-attn
├── action_head                       # Linear(hidden, 14)
└── is_pad_head                       # Linear(hidden, 1)
```

### 4.2 CVAE Encoder（训练分支）

```python
# L88-110：is_training = actions is not None
action_embed = encoder_action_proj(actions)     # (B, 100, hidden)
qpos_embed   = encoder_joint_proj(qpos)         # (B, 1, hidden)
cls_embed    = ...                              # (B, 1, hidden)
encoder_input = cat([CLS, qpos, actions], dim=1)  # (B, 102, hidden)

encoder_output = self.encoder(encoder_input, pos=pos_table, mask=is_pad)
latent_info = latent_proj(encoder_output[0])    # 取 CLS 位置
mu, logvar = latent_info[:, :32], latent_info[:, 32:]
z = reparametrize(mu, logvar)
latent_input = latent_out_proj(z)               # (B, hidden)
```

推理时（L111-114）：

```python
latent_sample = torch.zeros([bs, 32], ...)
latent_input = self.latent_out_proj(latent_sample)
```

### 4.3 Transformer Decoder（训练 + 推理）

```python
# L116-131
for cam_id in camera_names:
    features, pos = self.backbones[0](image[:, cam_id])  # ★ 共享 backbone
    all_cam_features.append(input_proj(features[0]))
    all_cam_pos.append(pos[0])

src = cat(all_cam_features, axis=3)    # 相机在 width 维拼接
pos = cat(all_cam_pos, axis=3)
proprio_input = input_proj_robot_state(qpos)

hs = self.transformer(
    src, None, query_embed.weight, pos,
    latent_input, proprio_input, additional_pos_embed.weight)[0]

a_hat = action_head(hs)                # (B, 100, 14)
```

`transformer.py` L49-77 会把 **latent token + proprio token** 与视觉 token 拼成 memory，100 个 **learned query** 做 cross-attention 解码。

### 4.4 Tensor 形状一览（B=batch, K=100, C=2 相机）

| 步骤 | Tensor | Shape |
|------|--------|-------|
| 输入 image | 多相机 RGB | `(B, C, 3, H, W)` |
| 输入 qpos | | `(B, 14)` |
| 训练 actions | future chunk | `(B, K, 14)` |
| ResNet / 相机 | feature map | `(B, 512, H', W')` |
| concat 相机 (width) | memory | `(B, hidden, H', W'·C)` |
| query_embed | | `(K, hidden)` |
| 输出 a_hat | chunk | `(B, K, 14)` |
| μ, logvar | CVAE | `(B, 32)` |

---

## 5. 第 4 层：视觉骨干 ResNet18

与 CNNMLP 共用 [`backbone.py`](../../../code/act/detr/models/backbone.py)，但用法不同：

| | ACT | CNNMLP |
|---|-----|--------|
| backbone 数量 | **1 个共享** | 每相机 1 个 |
| 特征用法 | `input_proj` → Transformer memory | 每相机 down_proj → flatten → MLP |
| 位置编码 | **sine pos** 进 cross-attn | 丢弃 `pos` |

Policy 层统一做 **ImageNet Normalize**；Dataset 只做 `/255.0`。

---

## 6. 第 5 层：数据从 HDF5 到 batch

**文件**：[`code/act/utils.py`](../../../code/act/utils.py) — 与 CNNMLP **同一 Dataset**。

### 6.1 ACT 实际用哪些字段

| 字段 | ACT 训练 | ACT 推理 |
|------|---------|---------|
| `image` | 当前帧 | 当前帧 |
| `qpos` | 当前 | 当前 |
| `action[:100]` | 监督标签 | — |
| `is_pad[:100]` | L1 mask | — |

随机 `start_ts` 采样 → 从该时刻起的 **整条 future** pad 到 `episode_len`，再截前 100 步。

含义：学的是 **「在 t 时刻看到 (image, qpos)，未来 100 步关节轨迹应是什么」** —— 显式 chunk BC，而非马尔可夫一步。

---

## 7. 第 6 层：评估时 chunk 如何执行

**文件**：`imitate_episodes.py` — `eval_bc`

### 7.1 无 temporal_agg（默认）

```text
t=0:    query → chunk[0..99]  → 执行 a[0]
t=1:    不 query               → 执行 a[1]
...
t=99:   不 query               → 执行 a[99]
t=100:  query → 新 chunk       → 执行 a[0]
```

```python
raw_action = all_actions[:, t % query_frequency]
```

开环执行 100 步后再重新观测 —— 减少 compounding，但 k 过大时会失去 reactive 能力（论文消融 k=200~400 略降）。

### 7.2 temporal_agg（`--temporal_agg`）

- 每步都 query，得到新 chunk
- 对 **同一时刻 t** 的多次预测存进 `all_time_actions[:, t]`
- 指数衰减权重 `exp(-0.01 * age)` 加权平均 → 动作更平滑
- 论文：ACT + TE 约 **+3.3%** 成功率

### 7.3 后处理

```python
action = raw_action * action_std + action_mean   # 反归一化
env.step(action)                                 # 14 维 absolute joint
```

---

## 8. 端到端流程图

### 训练

```text
episode.hdf5
    │
    ▼ EpisodicDataset：随机 start_ts
(image, qpos, action_pad, is_pad)
    │
    ▼ ACTPolicy(qpos, image, actions, is_pad)
    │     ├─ actions[:, :100], is_pad[:, :100]
    │     ├─ DETRVAE 训练路径
    │     │     ├─ Encoder: actions → z
    │     │     ├─ ResNet[0]: 多相机特征
    │     │     └─ Decoder: 100 queries → â (B,100,14)
    │     ├─ L1(â, a) with pad mask
    │     └─ KL(μ,logσ²) × kl_weight
    ▼
backward + AdamW
```

### 推理

```text
env.obs → (qpos_t, image_t)
    → ACTPolicy(qpos, image)     # 无 actions
    → DETRVAE 推理：z=0, 无 Encoder
    → chunk (1, 100, 14)
    → eval_bc 取 chunk[i] 或 temporal_agg
    → denormalize → env.step
```

---

## 9. ACTPolicy 相对 CNNMLP 解决了什么

| CNNMLP 痛点 | ACT 代码 / 论文对应 |
|-------------|-------------------|
| 误差累积 | chunk k=100；论文 k=1 → **1%**，k=100 → **44%** |
| 只监督下一步 | `actions[:, :100]` 整段 L1 |
| 多模态 demo 学平均 | CVAE Encoder + 推理 z=0；human demo 无 CVAE **2% vs 35%** |
| flatten+MLP 无 attention | Transformer cross-attn |
| 每步 query 算力高 | 默认每 **100** 步 query 一次 |
| 动作抖动 | 可选 `--temporal_agg` |

**仍存在的局限**（观测侧）：

- 推理时仍 **不看历史图像**（一阶马尔可夫 obs）
- chunk 内 **开环**，中途偏差不能 mid-chunk 修正（除非开 TE 每步 query）
- `kl_weight`、`chunk_size` 需调；k 过大 reactive 变差

---

## 10. 关键文件速查

| 顺序 | 文件 | 读什么 |
|:----:|------|--------|
| 1 | `imitate_episodes.py` | L53-68 配置；L191-261 eval chunk/TE；L316-319 训练 |
| 2 | `policy.py` | `ACTPolicy` + `kl_divergence` |
| 3 | `detr/main.py` | `build_ACT_model_and_optimizer` |
| 4 | `detr_vae.py` | `DETRVAE` + `build()` |
| 5 | `transformer.py` | Decoder cross-attn；latent/proprio 拼 memory |
| 6 | `backbone.py` | 共享 ResNet18 |
| 7 | `utils.py` | `EpisodicDataset` |

---

## 11. 自测

- [ ] 训练时 CVAE Encoder 是否运行？推理时呢？（**训练开 / 推理关**）
- [ ] 推理时 z 取什么？（**zeros(32)**）
- [ ] 输出 shape？（**(B, chunk_size, 14)**，默认 100）
- [ ] 损失组成？（**L1 + kl_weight × KL**）
- [ ] 默认每几步 query 一次 policy？（**100**；TE 模式下 **1**）
- [ ] `is_pad` 在 CNNMLP 和 ACT 里分别用不用？（CNNMLP **不用** / ACT **用于 L1 mask**）
- [ ] ACT 与 CNNMLP 的 backbone 共享方式？（ACT **共享** / CNNMLP **每相机独立**）

---

*上一篇：[CNNMLPPolicy 代码导读](./CNNMLPPolicy-Code-Walkthrough.md) · 原理：[ACT 模型工作原理](./ACT-Model-Working-Principles.md)*
