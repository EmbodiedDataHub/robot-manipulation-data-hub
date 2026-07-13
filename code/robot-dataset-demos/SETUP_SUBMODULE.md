# 首次发布为 GitHub 子仓库

本目录是独立 git 仓库，计划作为主仓库 submodule：

`git@github.com:EmbodiedDataHub/robot-dataset-demos.git`

## 1. 在 GitHub 创建空仓库

在 EmbodiedDataHub 组织下创建 **robot-dataset-demos**（Private/Public 均可，不要勾选 README）。

## 2. 推送本仓库

```bash
cd code/robot-dataset-demos
git remote add origin git@github.com:EmbodiedDataHub/robot-dataset-demos.git
git push -u origin main
```

## 3. 在主仓库注册 submodule

```bash
cd /path/to/robot-manipulation-data-hub

# 若曾存在旧目录 code/dataset-demos，先删除
rm -rf code/dataset-demos

# 若 code/robot-dataset-demos 已是普通目录，先移走再 submodule add
rm -rf code/robot-dataset-demos

git submodule add git@github.com:EmbodiedDataHub/robot-dataset-demos.git code/robot-dataset-demos
git commit -m "Add robot-dataset-demos submodule"
```

## 4. 克隆主仓库时拉子仓库

```bash
git clone --recurse-submodules git@github.com:EmbodiedDataHub/robot-manipulation-data-hub.git
# 或克隆后：
git submodule update --init code/robot-dataset-demos
```

## 5. 下载真实样例数据（可选，~67MB）

```bash
cd code/robot-dataset-demos
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
./scripts/download_samples.sh --lerobot-only
```

已下载的数据在 `samples/lerobot_aloha_sim_hf/`（gitignore，不提交）。
