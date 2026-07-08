---
title: "ACT 模型工作原理（代码导读）"
authors: "Tony Z. Zhao, Vikash Kumar, et al. (ALOHA)"
year: 2023
source: "code/act (Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware)"
tags:
  - ACT
  - imitation-learning
  - transformer
  - CVAE
  - action-chunking
  - ALOHA
  - robotics
aliases:
  - Action Chunking with Transformers
code_path: "code/act"
---

## 一句话总结

ACT（Action Chunking with Transformers）用 **CVAE + DETR-style Transformer** 一次预测未来一段动作序列（chunk），训练时从 demo 动作序列编码 latent，推理时用观测 + 零 latent 解码 chunk，逐步执行以完成双臂操作模仿。

---

## 1. 总体思路

ACT 不是逐步预测单步动作，而是：

1. 输入当前观测（关节位置 `qpos` + 多相机图像）
2. 一次输出未来 `chunk_size`（即 `num_queries`）步动作
3. 环境每步只执行 chunk 中的 **一个** 动作；每隔 `chunk_size` 步重新 query 一次（或用 temporal aggregation 每步都 query）

训练时采用 **Conditional VAE（CVAE）**：

- **Encoder**：用「未来动作序列 + 当前 qpos」编码出 latent `z`
- **Decoder**：用「当前观测 + z」解码出动作 chunk

推理时没有 ground-truth 动作，latent 固定为 **零向量**（先验均值），只靠观测解码动作。

### 1.1 通俗理解：chunk、query、训练时还在预测什么

#### chunk = 一小段「未来动作剧本」

不是只预测「下一步怎么动」，而是一次预测 **接下来 N 步**（默认 N=100）。这 N 步合在一起叫 **chunk（动作块）**。

```
普通策略：  现在 → 预测第 1 步 → 下一步再预测第 1 步 → …
ACT：       现在 → 预测 [第1步, 第2步, …, 第100步] = 一个 chunk
            环境仍每个控制周期只执行 1 步，其余步留到后面用
```

代码对应：`--chunk_size 100` → `policy_config['num_queries']` → `DETRVAE.num_queries`。

#### query = 调用一次模型，要一段新 chunk

**query** 不是 Transformer 里的 query embedding，而是工程用语：**问模型「接下来该怎么动？」**。

```python
# imitate_episodes.py — 这就是一次 query
all_actions = policy(qpos, curr_image)   # 输出 shape: (1, 100, 14)
```

| 术语 | 人话 | 代码 |
|------|------|------|
| chunk | 一次规划的一小段未来动作 | `chunk_size` / `num_queries` |
| query | 调用 `policy()` 要一段新 chunk | `t % query_frequency == 0` 时执行 |
| 逐步执行 | chunk 有 100 行，每次只执行 1 行 | `all_actions[:, t % query_frequency]` |

**执行时间线**（`chunk_size=100`）：

| 时刻 t | 是否 query | 执行的动作 |
|--------|-----------|-----------|
| 0 | ✅ | chunk 第 0 步 |
| 1~99 | ❌ 复用上次 chunk | chunk 第 1~99 步 |
| 100 | ✅ 重新 query | **新** chunk 第 0 步 |

#### 训练时输入了未来动作，还在预测什么？

训练时有专家 demo 的「标准答案」，但网络拆成 **两个角色**：

| 子网络 | 训练时能看什么 | 干什么 | 推理时 |
|--------|--------------|--------|--------|
| **Encoder**（CVAE 编码器） | qpos + **未来 demo 动作** | 把「这段打算怎么动」压成 32 维 `z` | **不运行**（没有答案） |
| **Decoder**（真正策略） | qpos + **图像** + `z` | 预测 100 步 chunk `a_hat` | 只看 qpos + 图像 + `z=0` |

**Decoder 预测的是**：在 **看不到** demo 动作序列的前提下，只靠图像和意图 `z`，去 **重建** 专家的那 100 步。

损失：`L1(a_hat, demo_actions)` —— 预测 chunk 要接近 demo chunk。

**考试类比**：

- Encoder = 看过答案后写出的「解题思路摘要」→ `z`
- Decoder = 学生闭卷写答案（看题目 + 思路摘要）
- 推理 = 没有摘要生成器，学生用默认思路 `z=0`，只看题目（图像）写答案

所以：**未来动作不是 Decoder 的输入，而是 Encoder 的输入 + Decoder 的监督标签。**

### 1.2 三个核心变量对照（action / 未来100步 demo / z）

| 名字 | 是什么 | shape | 代码里从哪来 |
|------|--------|-------|-------------|
| **action（单步）** | 专家 demo 里某一时刻的 **绝对关节目标**（14 维，不是 Δ 角） | `(14,)` | `hdf5['/action'][t]` |
| **未来100步 demo 动作** | 从随机 `start_ts` 起，demo 里连续 100 步 action 拼成的序列 | `(100, 14)` | `action[start_ts:]` 再截断到 `num_queries` |
| **z** | CVAE 的 **32 维隐变量**，整段轨迹意图的压缩，无直接物理含义 | `(32,)` | Encoder 输出；推理时 `torch.zeros(32)` |

---

## 1.3 带详细注释的完整源码（按调用链阅读）

> 以下均为 `code/act/` 中的 **真实代码**，只在行间加了中文注释，未改逻辑。  
> 建议顺序：`utils.py` → `policy.py` → `detr_vae.py` → `transformer.py` → `imitate_episodes.py`

---

### ① 数据：`EpisodicDataset.__getitem__`

**文件**：`code/act/utils.py`

```python
def __getitem__(self, index):
    sample_full_episode = False  # 始终 False：随机截一段，而不是整条 episode

    episode_id = self.episode_ids[index]
    dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        original_action_shape = root['/action'].shape   # 例如 (400, 14)：400 步，每步 14 维
        episode_len = original_action_shape[0]

        # ── 随机选「现在」是哪一帧 ──
        start_ts = np.random.choice(episode_len)        # 例如 start_ts=50

        # ── 观测：只取 start_ts 这一帧（和推理时「当前帧」对齐）──
        qpos = root['/observations/qpos'][start_ts]     # shape (14,) 当前关节角
        qvel = root['/observations/qvel'][start_ts]     # 读了但未使用
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

        # ── 监督：从 start_ts 到 episode 结束的「未来 demo 动作」──
        # 这就是「未来 demo 动作序列」的来源，可能 >100 步，后面会 pad / 截断
        if is_sim:
            action = root['/action'][start_ts:]         # action[50], action[51], ... action[399]
            action_len = episode_len - start_ts         # 还剩多少步真实动作，例如 350
        else:
            action = root['/action'][max(0, start_ts - 1):]
            action_len = episode_len - max(0, start_ts - 1)

    # ── pad 到固定长度 episode_len，末尾填 0，is_pad 标记假步 ──
    padded_action = np.zeros(original_action_shape, dtype=np.float32)  # (400, 14)
    padded_action[:action_len] = action           # 前 action_len 行是真实 demo，后面全是 0
    is_pad = np.zeros(episode_len)
    is_pad[action_len:] = 1                       # 末尾 pad 位置 = True

    all_cam_images = np.stack([image_dict[c] for c in self.camera_names], axis=0)

    image_data = torch.from_numpy(all_cam_images)
    qpos_data = torch.from_numpy(qpos).float()
    action_data = torch.from_numpy(padded_action).float()   # (episode_len, 14)，含 pad
    is_pad = torch.from_numpy(is_pad).bool()                  # (episode_len,)

    image_data = torch.einsum('k h w c -> k c h w', image_data)  # NCHW
    image_data = image_data / 255.0
    action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
    qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

    # 返回给 DataLoader；batch 后 shape：
    #   image:  (B, num_cam, 3, H, W)
    #   qpos:   (B, 14)
    #   action: (B, episode_len, 14)   ← 还会被 policy 截成 (B, 100, 14)
    #   is_pad: (B, episode_len)
    return image_data, qpos_data, action_data, is_pad
```

**`action[t]` 的物理含义**（`sim_env.py` 注释）：

```python
# Action space: [left_arm_qpos(6), left_gripper(1),
#               right_arm_qpos(6), right_gripper(1)]  ← 全是 absolute joint position
```

---

### ② 训练入口：`forward_pass`

**文件**：`code/act/imitate_episodes.py`

```python
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()    # demo 未来动作，含 pad
    is_pad = is_pad.cuda()
    # 传入 actions → ACTPolicy 走训练分支 → DETRVAE 走 Encoder+Decoder
    return policy(qpos_data, image_data, action_data, is_pad)
```

```python
# train_bc 里每个 batch：
forward_dict = forward_pass(data, policy)
loss = forward_dict['loss']             # l1 + kl_weight * kl
loss.backward()
optimizer.step()
```

---

### ③ 策略封装：`ACTPolicy.__call__` + KL

**文件**：`code/act/policy.py`

```python
class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model                # 实际是 DETRVAE（Encoder + Decoder 都在里面）
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']   # 默认 10

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        # ImageNet 归一化（在 /255 之后）；与预训练 ResNet 一致
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:  # ──────────── 训练 / 验证 ────────────
            # 只保留前 num_queries(=100) 步 → 这就是「未来100步 demo 动作」
            actions = actions[:, :self.model.num_queries]   # (B, 100, 14)
            is_pad  = is_pad[:, :self.model.num_queries]    # (B, 100)

            # 核心前向：actions 传进去 → DETRVAE 会跑 CVAE Encoder
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad)
            # a_hat: 模型预测的 chunk (B, 100, 14)
            # mu, logvar: VAE 参数，用于 KL

            total_kld, _, _ = kl_divergence(mu, logvar)

            # L1：逐步比较预测 vs demo，pad 步不参与
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')  # (B,100,14)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = dict()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict

        else:  # ──────────── 推理 ────────────
            # 不传 actions → DETRVAE 跳过 Encoder，z=0
            a_hat, _, (_, _) = self.model(qpos, image, env_state)
            return a_hat    # (B, 100, 14) 动作 chunk


def kl_divergence(mu, logvar):
    # 标准 VAE KL: KL(q(z|x) || N(0,I))
    # mu, logvar shape: (B, 32)
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, 32)
    total_kld = klds.sum(1).mean(0, True)   # 32 维求和，batch 平均 → 标量
    return total_kld, klds.mean(0), klds.mean(1).mean(0, True)
```

---

### ④ 核心网络：`DETRVAE.forward`（Encoder + Decoder）

**文件**：`code/act/detr/models/detr_vae.py`

```python
def reparametrize(mu, logvar):
    # VAE 重参数化：z = μ + σ·ε，ε~N(0,1)
    std = logvar.div(2).exp()              # σ = exp(logvar/2)
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps                  # z shape: (B, 32)


class DETRVAE(nn.Module):
    def __init__(self, ...):
        ...
        self.num_queries = num_queries       # 100，= chunk 长度
        self.latent_dim = 32                 # z 的维度：32 个抽象数，不是关节角
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 100 个动作 slot
        self.action_head = nn.Linear(hidden_dim, 14)              # 每个 slot → 14 维 action
        # CVAE Encoder 专用层：
        self.encoder = encoder               # 4 层 TransformerEncoder（与下面 self.transformer 不同！）
        self.encoder_action_proj = nn.Linear(14, hidden_dim)
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, 64)   # → μ(32) + logvar(32)
        self.latent_out_proj = nn.Linear(32, hidden_dim) # z → Decoder 可用的 token

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        is_training = actions is not None    # 有没有传 demo 动作
        bs, _ = qpos.shape

        # ═══════════════════════════════════════════════════════
        #  Part A：CVAE Encoder —— 仅训练时运行
        #  输入：qpos + 未来100步 demo 动作
        #  输出：32 维 z（再投影成 latent_input 给 Decoder）
        # ═══════════════════════════════════════════════════════
        if is_training:
            # 每一步 14 维 demo action → hidden_dim 维向量
            action_embed = self.encoder_action_proj(actions)  # (B, 100, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)        # (B, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (B, 1, hidden_dim)

            cls_embed = self.cls_embed.weight                 # (1, hidden_dim)
            cls_embed = cls_embed.unsqueeze(0).repeat(bs, 1, 1)  # (B, 1, hidden_dim)

            # 序列：[CLS, qpos, demo_a0, demo_a1, ..., demo_a99]  共 102 个 token
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)    # (102, B, hidden_dim) PyTorch MHA 格式

            # CLS 和 qpos 永远不 mask；demo 动作末尾 pad 步 mask 掉
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (B, 102)

            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)

            # CVAE Encoder：自注意力读完整 demo 序列
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]    # 只取 CLS 位置 → (B, hidden_dim)

            # 投影到 VAE 的 μ 和 log σ²
            latent_info = self.latent_proj(encoder_output)       # (B, 64)
            mu = latent_info[:, :self.latent_dim]                # (B, 32)
            logvar = latent_info[:, self.latent_dim:]            # (B, 32)
            latent_sample = reparametrize(mu, logvar)            # z (B, 32) ← 这就是 z
            latent_input = self.latent_out_proj(latent_sample)   # (B, hidden_dim) → 给 Decoder

        else:
            # ═══ 推理：没有 demo 动作，Encoder 整个跳过 ═══
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            # z 全 0 = 标准正态先验的均值（不是关节角！）
            latent_input = self.latent_out_proj(latent_sample)   # (B, hidden_dim)

        # ═══════════════════════════════════════════════════════
        #  Part B：Decoder —— 训练 / 推理都运行
        #  输入：当前 qpos + 当前 image + latent_input（来自 z）
        #  输出：a_hat (B, 100, 14) 预测的 chunk
        # ═══════════════════════════════════════════════════════
        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                # ResNet18 提特征；多相机共用同一个 backbone（代码 HARDCODED）
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]           # 最后一层 feature map
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))  # 1×1 conv → hidden_dim
                all_cam_pos.append(pos)

            proprio_input = self.input_proj_robot_state(qpos)  # (B, hidden_dim)

            # 多相机 feature map 在宽度维拼接
            src = torch.cat(all_cam_features, axis=3)   # (B, hidden_dim, H, W_total)
            pos = torch.cat(all_cam_pos, axis=3)

            # 主 Transformer：100 个 query slot → 100 步 action
            hs = self.transformer(
                src, None,
                self.query_embed.weight,              # (100, hidden_dim) 100 个动作槽
                pos,
                latent_input,                         # 来自 z
                proprio_input,                        # 来自 qpos
                self.additional_pos_embed.weight
            )[0]                                      # (1, B, 100, hidden_dim)

        a_hat = self.action_head(hs)                  # (B, 100, 14) ← 最终预测的 chunk
        is_pad_hat = self.is_pad_head(hs)             # 未参与 loss
        return a_hat, is_pad_hat, [mu, logvar]
```

---

### ⑤ Transformer Decoder：100 个 query 如何变成 chunk

**文件**：`code/act/detr/models/transformer.py`

```python
def forward(self, src, mask, query_embed, pos_embed,
            latent_input=None, proprio_input=None, additional_pos_embed=None):

    if len(src.shape) == 4:  # src = 图像 feature map (B, C, H, W)
        bs, c, h, w = src.shape
        # 把每个空间位置变成一个 token
        src = src.flatten(2).permute(2, 0, 1)           # (H*W, B, hidden_dim)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (100, B, hidden_dim)

        additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)
        pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

        # 在图像 token 序列最前面插入 2 个 token：[z, qpos]
        addition_input = torch.stack([latent_input, proprio_input], axis=0)  # (2, B, hidden_dim)
        src = torch.cat([addition_input, src], axis=0)
        # memory 序列 = [z_token, qpos_token, pix_0, pix_1, ..., pix_{H*W-1}]

    # Decoder 的 100 个 slot：内容初始化为 0，位置信息来自 query_embed
    tgt = torch.zeros_like(query_embed)               # (100, B, hidden_dim)

    # Encoder 再编码一遍 memory（注意：这是 self.transformer.encoder，不是 CVAE 的 self.encoder）
    memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

    # Decoder：100 个 query cross-attend 到 memory，每个 slot 读图像+z+qpos
    hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                      pos=pos_embed, query_pos=query_embed)
    hs = hs.transpose(1, 2)                         # (1, B, 100, hidden_dim)
    return hs
    # 回到 detr_vae.py：action_head(hs) → (B, 100, 14)
    # query_embed[i] 对应 chunk 里第 i 步的预测
```

---

### ⑥ 推理闭环：chunk 怎么一步步执行

**文件**：`code/act/imitate_episodes.py`

```python
query_frequency = policy_config['num_queries']   # 100：每 100 步才重新调用 policy

for t in range(max_timesteps):
    qpos_numpy = np.array(obs['qpos'])
    qpos = (qpos_numpy - stats['qpos_mean']) / stats['qpos_std']
    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)   # (1, 14)
    curr_image = get_image(ts, camera_names)                     # (1, cam, 3, H, W)

    if config['policy_class'] == "ACT":
        if t % query_frequency == 0:
            # ── 一次 query：不传 actions → 推理分支 → 得到 chunk ──
            all_actions = policy(qpos, curr_image)   # (1, 100, 14)
            # all_actions[0, 0] = 预测的第 0 步
            # all_actions[0, 1] = 预测的第 1 步
            # ...
            # all_actions[0, 99] = 预测的第 99 步

        # 从 chunk 里取「当前时刻该执行的那一步」
        raw_action = all_actions[:, t % query_frequency]   # t=0→第0步, t=1→第1步, ..., t=100→重新query后第0步

    raw_action = raw_action.squeeze(0).cpu().numpy()       # (14,)
    action = raw_action * stats['action_std'] + stats['action_mean']  # 反归一化 → 真实关节角
    ts = env.step(action)   # 发给仿真/真机，只执行 1 步
```

**时间线**（`chunk_size=100`）：

```
t=0:   policy() → all_actions shape (1,100,14)  执行 all_actions[:,0]
t=1:   不调用 policy                            执行 all_actions[:,1]
...
t=99:  不调用 policy                            执行 all_actions[:,99]
t=100: policy() → 新的 all_actions              执行 all_actions[:,0]
```

---

### ⑦ 通读检查：对照代码回答

| 问题 | 答案 | 看哪行 |
|------|------|--------|
| 「未来100步 demo 动作」从哪来？ | `hdf5 action[start_ts:]`，再 `[:, :100]` 截断 | `utils.py` L44, `policy.py` L24 |
| 每步 action 是相对角还是绝对角？ | **绝对关节目标** 14 维 | `sim_env.py` L23 |
| z 是什么？ | **32 维 VAE 隐变量**，不是关节量 | `detr_vae.py` L67, L109 |
| 推理时 z 是多少？ | `torch.zeros(32)` | `detr_vae.py` L113 |
| chunk 在哪输出？ | `action_head(hs)` → `(B,100,14)` | `detr_vae.py` L137 |
| 环境每步执行 chunk 哪一步？ | `all_actions[:, t % 100]` | `imitate_episodes.py` L261 |

---

## 2. 数据维度与预处理

### 2.1 状态 / 动作维度

| 变量 | Shape | 含义 |
|------|-------|------|
| `qpos` | `(B, 14)` | 双臂 puppet 机器人 14 维关节位置（每臂 7 维，含夹爪） |
| `action` | `(B, T, 14)` | 从当前时刻起的未来动作序列，与 qpos 同维 |
| `image` | `(B, num_cam, 3, H, W)` | 多相机 RGB，值域 `[0, 1]` |
| `is_pad` | `(B, T)` | padding mask，`True` 表示该步是 pad（episode 末尾之后） |

`state_dim = 14` 在 `code/act/detr/models/detr_vae.py` 中硬编码。

### 2.2 训练样本构造（`code/act/utils.py` → `EpisodicDataset`）

每个样本从一条 demo episode 中随机采样起始时刻 `start_ts`：

```python
# 观测：仅 start_ts 时刻
qpos = root['/observations/qpos'][start_ts]
images = root[f'/observations/images/{cam_name}'][start_ts]

# 监督信号：从 start_ts 到 episode 结束的全部 action
action = root['/action'][start_ts:]
# 不足部分 pad 到 episode_len，is_pad 标记 pad 区域
```

归一化：

- `qpos`、`action`：减均值除标准差（全数据集统计）
- `image`：`/255.0`，模型内再做 ImageNet normalize

---

## 3. 模型结构（`DETRVAE`）

整体是 **CVAE + DETR-style Transformer Decoder**，定义在 `code/act/detr/models/detr_vae.py`。

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段                                   │
│  actions + qpos ──► CVAE Encoder ──► z (32维) ──┐               │
│                                                  │               │
│  qpos + images ─────────────────────────────────►│──► Decoder   │
│                                                  │      │        │
│                                                  └──► 动作 chunk │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        推理阶段                                   │
│  z = 0 (先验均值) ──────────────────────────────► Decoder       │
│  qpos + images ─────────────────────────────────►     │        │
│                                                       ▼        │
│                                              动作 chunk (T×14)  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 CVAE Encoder（动作序列编码器）

**独立** 于图像 Transformer，专门把「未来要做什么」压缩成 latent。

| 组件 | 说明 |
|------|------|
| `encoder_action_proj` | `Linear(14 → hidden_dim)`，逐步动作 embedding |
| `encoder_joint_proj` | `Linear(14 → hidden_dim)`，当前 qpos embedding |
| `cls_embed` | 可学习 CLS token |
| `pos_table` | 正弦位置编码，长度 `1 + 1 + num_queries` |
| `encoder` | 4 层 Transformer Encoder（`enc_layers=4`） |
| `latent_proj` | `Linear(hidden_dim → 64)`，输出 `μ` 和 `log σ²` 各 32 维 |
| `reparametrize` | `z = μ + σ·ε`，ε ~ N(0,I) |

输入序列拼接顺序：

```
[CLS] + [qpos] + [a_t, a_{t+1}, ..., a_{t+chunk-1}]
         ↑              ↑
      1 token      num_queries tokens（pad 步会被 mask）
```

Encoder 只取 **CLS 输出** → 投影得到 `(μ, logvar)` → 重参数化采样 `z` → `latent_out_proj` 映射到 `hidden_dim`。

### 3.2 CVAE Decoder（观测条件解码器）

基于 DETR 改造，用 **object query** 预测动作 chunk 的每一步。

| 组件 | 说明 |
|------|------|
| `backbone` | ResNet18（`lr_backbone=1e-5` 单独学习率） |
| `input_proj` | `Conv2d(C → hidden_dim)`，图像特征投影 |
| `input_proj_robot_state` | `Linear(14 → hidden_dim)`，本体感知 |
| `query_embed` | `Embedding(num_queries, hidden_dim)`，可学习 query，一步 query 对应一步未来动作 |
| `transformer` | Encoder 4 层 + Decoder 7 层（默认 `dec_layers=7`） |
| `action_head` | `Linear(hidden_dim → 14)` |
| `is_pad_head` | `Linear(hidden_dim → 1)`，预测 pad（**当前 loss 未使用**） |

Decoder memory 拼接顺序（`code/act/detr/models/transformer.py`）：

```
memory = [latent_input, proprio_input, image_features...]
          ↑ 2个额外 token    ↑ ResNet 特征图 flatten 后的 spatial tokens
```

Decoder 用 `num_queries` 个零初始化 target + 可学习 `query_embed` 做 cross-attention，输出：

```
hs: (1, B, num_queries, hidden_dim)  →  action_head  →  a_hat: (B, num_queries, 14)
```

### 3.3 默认超参（`code/act/imitate_episodes.py`）

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `chunk_size` / `num_queries` | 100 | 一次预测的未来步数 |
| `hidden_dim` | 512 | Transformer 隐层维度 |
| `dim_feedforward` | 3200 | FFN 中间层 |
| `nheads` | 8 | 注意力头数 |
| `enc_layers` | 4 | CVAE encoder 层数 |
| `dec_layers` | 7 | 主 Transformer decoder 层数 |
| `latent_dim` | 32 | VAE latent 维度 |
| `kl_weight` | 10 | KL 散度权重 |

---

## 4. 训练阶段

### 4.1 输入

`ACTPolicy.__call__(qpos, image, actions, is_pad)`（`code/act/policy.py`）：

| 输入 | Shape | 来源 |
|------|-------|------|
| `qpos` | `(B, 14)` | 随机时刻的关节位置（已归一化） |
| `image` | `(B, num_cam, 3, H, W)` | 同一时刻多相机图像 |
| `actions` | `(B, episode_len, 14)` | 从该时刻起的未来动作（pad 后） |
| `is_pad` | `(B, episode_len)` | padding mask |

进入模型前还会：

- 图像 ImageNet normalize
- `actions`、`is_pad` 截断到前 `num_queries` 步

### 4.2 前向流程

```python
# policy.py 训练分支
a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
```

1. **Encoder 路径**：`actions + qpos` → Transformer Encoder → `(μ, logvar)` → 采样 `z`
2. **Decoder 路径**：`qpos + image + z` → ResNet + Transformer → `a_hat`

### 4.3 输出

`ACTPolicy` 返回 **loss 字典**，不是动作：

| 键 | 含义 |
|----|------|
| `l1` | 动作 L1 重建损失（mask 掉 pad） |
| `kl` | KL(q(z\|a,q) \| N(0,I)) |
| `loss` | `l1 + kl_weight * kl` |

训练循环（`code/act/imitate_episodes.py`）：

```python
forward_dict = policy(qpos_data, image_data, action_data, is_pad)
loss = forward_dict['loss']
loss.backward()
optimizer.step()
```

---

## 5. 推理阶段

### 5.1 输入

`ACTPolicy.__call__(qpos, image)` —— **不传 actions**：

| 输入 | Shape | 说明 |
|------|-------|------|
| `qpos` | `(1, 14)` | 当前关节位置（归一化） |
| `image` | `(1, num_cam, 3, H, W)` | 当前多相机图像 |

### 5.2 前向流程

```python
# detr_vae.py 推理分支
latent_sample = torch.zeros([bs, latent_dim])  # z = 0，不采样
latent_input = self.latent_out_proj(latent_sample)
# 仅走 Decoder：qpos + image + z=0 → Transformer → 动作 chunk
```

### 5.3 输出

| 输出 | Shape | 说明 |
|------|-------|------|
| `a_hat` | `(1, num_queries, 14)` | 未来 `chunk_size` 步动作（归一化空间） |

推理后需 **反归一化** 再发给机器人：

```python
action = a_hat * stats['action_std'] + stats['action_mean']
```

### 5.4 如何在环境中逐步执行（`eval_bc`）

两种模式：

**A. 标准 chunking（默认）**

```python
query_frequency = num_queries  # 例如 100
if t % query_frequency == 0:
    all_actions = policy(qpos, curr_image)  # 重新预测整段 chunk
raw_action = all_actions[:, t % query_frequency]  # 取 chunk 内对应步
```

每 100 步 query 一次，中间 99 步复用同一段预测。

**B. Temporal Aggregation（`--temporal_agg`）**

每步都 query，对历史所有预测做指数加权平均：

```python
query_frequency = 1
all_time_actions[t, t:t+num_queries] = all_actions
# 对时刻 t 的所有历史预测加权平均
exp_weights = exp(-0.01 * arange(n))
raw_action = weighted_sum(actions_for_curr_step)
```

更平滑，但计算量更大。

---

## 6. 损失函数设计

定义在 `code/act/policy.py`：

### 6.1 L1 重建损失（主损失）

```python
all_l1 = F.l1_loss(actions, a_hat, reduction='none')  # (B, T, 14)
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
```

- 逐步 L1，只对 **非 pad** 步计算
- 14 维动作等权平均
- 选择 L1 而非 L2，对 outlier 更鲁棒，轨迹更平滑

### 6.2 KL 散度（VAE 正则）

```python
klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
total_kld = klds.sum(1).mean(0, True)  # 对 latent 32 维求和，再 batch 平均
```

标准高斯 VAE：`KL(q(z|x) || N(0, I))`

### 6.3 总损失

```python
loss = l1 + kl_weight * kl   # 默认 kl_weight = 10
```

| 项 | 作用 |
|----|------|
| L1 | 让 decoder 学会从观测 + z 重建动作 chunk |
| KL | 约束 latent 分布接近标准正态，使推理时 z=0 可用 |
| `kl_weight` | 权衡重建精度 vs latent 正则；过大动作更「平均」，过小推理时 z=0 效果差 |

### 6.4 未使用的 head

`is_pad_head` 会输出 pad 预测，但 **当前 loss 没有监督它**，实际只用 `is_pad` 做 L1 mask。

---

## 7. 端到端数据流对比

### 训练

```
HDF5 episode
    │
    ├─ 随机 start_ts
    │
    ▼
┌──────────────────────────────────────────┐
│ qpos (B,14)          actions (B,T,14)    │
│ image (B,cam,3,H,W)  is_pad (B,T)        │
└──────────────────────────────────────────┘
    │
    ▼ ACTPolicy (训练模式)
    │
    ├─ Encoder: [CLS,qpos,actions] → μ,logvar → z
    │
    └─ Decoder: [z, qpos, image_feat] → a_hat (B,T,14)
    │
    ▼
loss = masked_L1(actions, a_hat) + kl_weight * KL(μ,logvar)
```

### 推理

```
环境观测 (qpos, images)
    │
    ▼ 归一化 + ImageNet normalize
    │
    ▼ ACTPolicy (推理模式, z=0)
    │
    ▼ a_hat (1, chunk_size, 14)
    │
    ▼ 反归一化 → 取当前步 action → env.step()
    │
    └── 每 chunk_size 步（或每步 + temporal agg）重新 query
```

---

## 8. 与 CNNMLP baseline 的对比

同仓库还提供 `CNNMLPPolicy`（`code/act/policy.py`）作为 baseline：

| | ACT | CNNMLP |
|---|-----|--------|
| 预测步数 | `chunk_size`（如 100） | 1 |
| 结构 | CVAE + Transformer | ResNet + MLP |
| 损失 | L1 + KL | MSE |
| 多模态 | 有（VAE latent） | 无 |

ACT 的核心优势在于 **action chunking**：一次规划多步，减少 compounding error，并通过 temporal aggregation 进一步平滑。

---

## 10. 代码全流程导读（按执行顺序读代码）

> 建议打开以下文件对照阅读：`imitate_episodes.py` → `utils.py` → `policy.py` → `detr_vae.py` → `transformer.py`

### 10.0 代码文件地图

```
imitate_episodes.py     训练/评估主入口，chunk 执行逻辑
    ├── utils.py        数据集 EpisodicDataset，归一化
    ├── policy.py       ACTPolicy 封装，损失计算，训练/推理分叉
    └── detr/
        ├── main.py     build_ACT_model_and_optimizer
        └── models/
            ├── detr_vae.py    DETRVAE 核心网络（Encoder + Decoder）
            ├── transformer.py DETR Transformer（Decoder 侧）
            └── backbone.py    ResNet18 图像特征
```

**一张图串全流程**：

```
HDF5 demo
  → EpisodicDataset.__getitem__     # 造 (image, qpos, action, is_pad)
  → forward_pass / eval_bc          # 喂给 policy
  → ACTPolicy.__call__              # 归一化图像，算 loss 或返回 a_hat
  → DETRVAE.forward                 # Encoder(训练) + Decoder(始终)
  → action_head                     # (B, 100, 14) chunk
```

---

### 10.1 入口：超参与模型构建

**文件**：`code/act/imitate_episodes.py`

命令行 `--chunk_size 100` 被映射为 `num_queries`，这是 chunk 长度的唯一来源：

```python
# imitate_episodes.py L53-68
if policy_class == 'ACT':
    policy_config = {
        'num_queries': args['chunk_size'],   # chunk 长度 = 100
        'kl_weight': args['kl_weight'],
        'hidden_dim': args['hidden_dim'],
        'enc_layers': 4,    # CVAE Encoder 层数
        'dec_layers': 7,    # 主 Transformer Decoder 层数
        ...
    }
```

**文件**：`code/act/detr/main.py` → `build_ACT_model_and_optimizer`

```python
# main.py L70-90
model = build_ACT_model(args)          # → detr_vae.build() → DETRVAE
optimizer = AdamW([
    {非 backbone 参数, lr=1e-5},
    {backbone 参数,   lr=1e-5},        # ResNet 用小学习率
])
```

**文件**：`code/act/detr/models/detr_vae.py` → `build()`

```python
# detr_vae.py L229-255
backbone = build_backbone(args)        # ResNet18
transformer = build_transformer(args)  # 4 enc + 7 dec 层
encoder = build_encoder(args)          # CVAE 专用 4 层 Encoder（独立于上面）
model = DETRVAE(backbones, transformer, encoder,
                state_dim=14, num_queries=args.num_queries, ...)
```

注意：代码里有 **两个** Transformer Encoder：
1. `self.encoder`（CVAE 侧）：编码动作序列 → `z`
2. `self.transformer.encoder`（Decoder 侧）：编码图像 memory

---

### 10.2 数据：一个训练样本长什么样

**文件**：`code/act/utils.py` → `EpisodicDataset.__getitem__`

```python
# utils.py L35-48 — 随机选一个起始帧
start_ts = np.random.choice(episode_len)

# 观测：只有这一帧
qpos = root['/observations/qpos'][start_ts]
image  = root[f'/observations/images/{cam_name}'][start_ts]

# 标签：从这一帧起的「未来所有动作」（即 chunk 的监督，可能超过 100 步）
action = root['/action'][start_ts:]
```

```python
# utils.py L51-54 — episode 末尾 pad 到固定长度
padded_action = np.zeros(original_action_shape)   # (episode_len, 14)
padded_action[:action_len] = action
is_pad = np.zeros(episode_len); is_pad[action_len:] = 1   # 末尾标 True
```

```python
# utils.py L68-74 — 归一化
image_data  = image / 255.0
action_data = (action - action_mean) / action_std
qpos_data   = (qpos   - qpos_mean)   / qpos_std

return image_data, qpos_data, action_data, is_pad
# shapes: (cam,3,H,W), (14,), (episode_len,14), (episode_len,)
```

**要点**：一条样本 = 「某一帧的观测」+ 「从该帧起的未来动作序列」。与推理时「当前观测 → 预测未来 chunk」对齐。

---

### 10.3 训练循环：数据怎么进网络

**文件**：`code/act/imitate_episodes.py`

```python
# L316-319
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    return policy(qpos_data, image_data, action_data, is_pad)  # 有 actions → 训练模式
```

```python
# L364-370
forward_dict = forward_pass(data, policy)
loss = forward_dict['loss']   # l1 + kl_weight * kl
loss.backward()
optimizer.step()
```

---

### 10.4 ACTPolicy：训练 / 推理分叉（第一个分叉点）

**文件**：`code/act/policy.py`

```python
# L18-38 — 用 actions is not None 区分训练/推理
def __call__(self, qpos, image, actions=None, is_pad=None):
    image = Normalize(ImageNet mean/std)(image)   # 在 /255 之后再 normalize

    if actions is not None:          # ========== 训练 ==========
        actions = actions[:, :self.model.num_queries]   # 截到前 100 步
        is_pad  = is_pad[:, :self.model.num_queries]

        a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, None, actions, is_pad)
        l1 = masked_L1(actions, a_hat, is_pad)
        kl = kl_divergence(mu, logvar)
        return {'l1': l1, 'kl': kl, 'loss': l1 + kl_weight * kl}

    else:                            # ========== 推理 ==========
        a_hat, _, _ = self.model(qpos, image, None)   # 不传 actions
        return a_hat    # (B, 100, 14)
```

**读代码技巧**：`actions is not None` 一路传到 `DETRVAE.forward`，触发 Encoder 分支。

---

### 10.5 DETRVAE.__init__：网络里有哪些层

**文件**：`code/act/detr/models/detr_vae.py` L34-76

| 模块 | 代码变量 | 作用 |
|------|---------|------|
| CVAE Encoder | `self.encoder` | 4 层 TransformerEncoder，读动作序列 |
| | `encoder_action_proj` | 每步 14 维 action → hidden_dim |
| | `encoder_joint_proj` | qpos 14 维 → hidden_dim |
| | `cls_embed` | 可学习 CLS token |
| | `latent_proj` | CLS 输出 → μ(32) + logvar(32) |
| | `latent_out_proj` | z(32) → hidden_dim，送给 Decoder |
| 图像 backbone | `self.backbones` | ResNet18 |
| | `input_proj` | 512 通道 → hidden_dim 的 1×1 卷积 |
| | `input_proj_robot_state` | qpos → hidden_dim |
| 主 Transformer | `self.transformer` | enc 4 层 + dec 7 层 |
| | `query_embed` | Embedding(100, hidden_dim)，**每个 slot 对应 chunk 里的一步** |
| 输出 | `action_head` | hidden_dim → 14 维关节动作 |
| | `is_pad_head` | 未参与 loss |

```python
# L54 — query_embed 就是 DETR 的 object query，这里 repurposed 为「第 k 步未来动作槽位」
self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 100 个 slot
self.action_head = nn.Linear(hidden_dim, state_dim)       # 每个 slot → 14 维 action
```

---

### 10.6 DETRVAE.forward 上半：CVAE Encoder（仅训练，L88-110）

```python
# detr_vae.py L85-86
is_training = actions is not None

if is_training:
    # Step 1: 把动作序列投影到 hidden_dim
    action_embed = self.encoder_action_proj(actions)     # (B, 100, 512)
    qpos_embed   = self.encoder_joint_proj(qpos).unsqueeze(1)  # (B, 1, 512)

    # Step 2: 拼序列 [CLS, qpos, a_0, a_1, ..., a_99]
    cls_embed = self.cls_embed.weight.expand(B, 1, -1)
    encoder_input = cat([cls_embed, qpos_embed, action_embed], dim=1)
    encoder_input = encoder_input.permute(1, 0, 2)       # (102, B, 512)  Transformer 格式

    # Step 3: padding mask — CLS 和 qpos 永不 mask，动作末尾 pad 步 mask 掉
    is_pad = cat([False, False, is_pad], dim=1)

    # Step 4: 过 CVAE Encoder，只取 CLS 位置输出
    encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
    encoder_output = encoder_output[0]                   # (B, 512)  第 0 个 token = CLS

    # Step 5: VAE —— 出 μ, logvar，重参数化采样 z
    latent_info = self.latent_proj(encoder_output)       # (B, 64)
    mu, logvar = latent_info[:, :32], latent_info[:, 32:]
    latent_sample = mu + exp(logvar/2) * eps             # (B, 32)
    latent_input = self.latent_out_proj(latent_sample)   # (B, 512) → 送给 Decoder
```

**推理分支**（L111-114）：

```python
else:
    latent_sample = torch.zeros([bs, 32])    # z = 0，不用 Encoder
    latent_input = self.latent_out_proj(latent_sample)
```

---

### 10.7 DETRVAE.forward 下半：Decoder 图像路径（L116-138）

```python
# Step 1: 每个相机过 ResNet18（代码里多相机共用 backbones[0]）
for cam_id in range(num_cameras):
    features, pos = self.backbones[0](image[:, cam_id])
    features = self.input_proj(features[0])   # 最后一层 feature map → hidden_dim
    all_cam_features.append(features)

# Step 2: 多相机特征在宽度维拼接
src = torch.cat(all_cam_features, axis=3)       # (B, 512, H, W_total)
pos = torch.cat(all_cam_pos, axis=3)

# Step 3: qpos 投影
proprio_input = self.input_proj_robot_state(qpos)  # (B, 512)

# Step 4: 进主 Transformer Decoder
hs = self.transformer(
    src, None,
    self.query_embed.weight,    # (100, 512) — 100 个动作 slot
    pos,
    latent_input,               # (B, 512) — 来自 z
    proprio_input,              # (B, 512) — 来自 qpos
    self.additional_pos_embed.weight
)[0]

# Step 5: 每个 query slot 回归一步动作
a_hat = self.action_head(hs)    # hs: (1, B, 100, 512) → a_hat: (B, 100, 14)
```

**shape 追踪**（`hidden_dim=512, chunk=100`）：

| 步骤 | Tensor | Shape |
|------|--------|-------|
| ResNet 输出 | features | `(B, 512, H, W)` |
| 拼 latent+qpos+图像 | memory src | `(2+H×W, B, 512)` |
| query slots | query_embed | `(100, B, 512)` |
| Decoder 输出 | hs | `(1, B, 100, 512)` |
| 动作 chunk | a_hat | `(B, 100, 14)` |

---

### 10.8 Transformer：memory 怎么拼、chunk 怎么出来

**文件**：`code/act/detr/models/transformer.py` L49-77

```python
def forward(self, src, mask, query_embed, pos_embed,
            latent_input=None, proprio_input=None, additional_pos_embed=None):

    # 图像 feature map 展平为 token 序列
    src = src.flatten(2).permute(2, 0, 1)          # (H*W, B, 512)

    # 在 memory 最前面插入 latent 和 proprio 两个 token
    addition_input = stack([latent_input, proprio_input], dim=0)  # (2, B, 512)
    src = cat([addition_input, src], dim=0)        # memory 序列

    # Decoder 的 100 个 query：初始为全零，靠 query_embed 提供位置信息
    tgt = zeros_like(query_embed)                  # (100, B, 512)

    memory = self.encoder(src, pos=pos_embed)      # 自注意力编码 memory
    hs = self.decoder(tgt, memory, query_pos=query_embed)  # cross-attn: query 读 memory
    return hs.transpose(1, 2)                    # (1, B, 100, 512)
```

**和 DETR 检测的类比**：

| DETR 原版 | ACT |
|-----------|-----|
| object query | `query_embed`（100 个 slot） |
| 每个 query 预测一个框 | 每个 query 预测一步 action (14 维) |
| image feature map = memory | ResNet feature + latent + qpos = memory |

**为何叫 num_queries**：DETR 遗产 —— Decoder 里 **100 个可学习 query slot**，每个 slot cross-attend 到图像 memory，输出一步动作。工程上的「query 模型」= 调用整个 `policy()`，是不同层面的含义。

---

### 10.9 损失函数：代码怎么算

**文件**：`code/act/policy.py` L27-34, L71-84

```python
# L1 重建 —— 只对非 pad 步
all_l1 = F.l1_loss(actions, a_hat, reduction='none')     # (B, 100, 14)
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

# KL —— 标准 VAE
klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())    # (B, 32)
total_kld = klds.sum(1).mean()

loss = l1 + kl_weight * total_kld    # 默认 kl_weight=10
```

---

### 10.10 推理闭环：eval_bc 里 chunk 怎么执行

**文件**：`code/act/imitate_episodes.py` L191-261

```python
query_frequency = policy_config['num_queries']   # 默认 100：每 100 步 query 一次

for t in range(max_timesteps):
    qpos = normalize(obs['qpos'])
    curr_image = get_image(ts, camera_names)

    # ===== 一次 query，得到一个 chunk =====
    if t % query_frequency == 0:
        all_actions = policy(qpos, curr_image)   # (1, 100, 14)，推理不传 actions

    # ===== 从 chunk 里取当前该执行的那一步 =====
    raw_action = all_actions[:, t % query_frequency]   # (1, 14)

    action = denormalize(raw_action)
    ts = env.step(action)                            # 只执行 1 步
```

**Temporal Aggregation**（`--temporal_agg`，L192-259）：

```python
query_frequency = 1                    # 每步都 query
all_time_actions[t, t:t+100] = all_actions   # 记录「在 t 时刻对未来 100 步的预测」
# 对「所有历史预测里对当前时刻 t 的预测」做指数加权平均
raw_action = weighted_sum(actions_for_curr_step)
```

---

### 10.11 通读检查清单

读完后应能回答：

- [ ] `chunk_size` 在代码哪一行变成 `num_queries`？→ `imitate_episodes.py` L58
- [ ] 训练样本的 action 从哪一帧开始切？→ `utils.py` L44 `action[start_ts:]`
- [ ] 什么条件触发 CVAE Encoder？→ `actions is not None`（`detr_vae.py` L85）
- [ ] 推理时 z 是多少？→ `torch.zeros(32)`（L113）
- [ ] 100 步 chunk 是哪一层输出的？→ `query_embed`(100 slot) → `action_head` → `(B,100,14)`
- [ ] 环境每步执行 chunk 的哪一步？→ `all_actions[:, t % query_frequency]`
- [ ] 总 loss 公式？→ `masked_L1 + kl_weight * KL`

---

## 11. 关键代码索引

| 功能 | 文件 |
|------|------|
| Policy 封装 & Loss | `code/act/policy.py` |
| CVAE 模型定义 | `code/act/detr/models/detr_vae.py` |
| Transformer | `code/act/detr/models/transformer.py` |
| 数据加载 | `code/act/utils.py` |
| 训练 / 评估循环 | `code/act/imitate_episodes.py` |
| 模型构建 & 优化器 | `code/act/detr/main.py` |

---

## 12. 小结

| 阶段 | 输入 | 输出 |
|------|------|------|
| **训练** | `qpos`, `image`, `actions`, `is_pad` | `loss_dict`（`l1`, `kl`, `loss`） |
| **推理** | `qpos`, `image` | `a_hat` `(B, chunk_size, 14)` 动作 chunk |

**模型本质**：以 DETR query 机制做 action chunking 的条件 VAE —— Encoder 从「未来动作 + 当前状态」学 latent，Decoder 从「当前观测 + z」解码未来一段轨迹；推理时用 z=0 作为先验均值，配合 chunk 执行或 temporal aggregation 完成闭环控制。
