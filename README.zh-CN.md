# Latent Bridge

WAN / V-RAE 视频 latent -> Bridge -> 冻结的 VGGT-Omega -> 49 帧深度与点云。

[English](README.md) | [数据格式](docs/DATA.md) | [评估协议](docs/EVALUATION.md)

本版提供 Bridge 训练和几何推理，不包含视频生成、预训练权重或数据。
许可证：[Apache-2.0](LICENSE)。

![Latent Bridge：对齐训练、生成推理及两种编码器对应的 Bridge](assets/method-overview.png)

**方法概览：** 训练时仅更新 Bridge，将视频 latent 对齐到冻结的 Omega 图像特征。
推理时，对齐后的 tokens 直接解码为深度、相机和点云，RGB 视频解码是独立分支。
图中生成模型不包含在本代码包内。

各入口脚本直接定义自己的参数并调用对应函数，没有统一命令分发层。
训练入口是 `train.py`，训练循环是 `latent_bridge/train.py`；网络在
`latent_bridge/models/`，推理流程在 `infer.py`。源码运行不需要手动设置导入路径。

## 1. 配置环境

在仓库根目录执行。要求 Python 3.10+、适配 GPU 的 PyTorch 2.6+。
以下命令使用 Python 3.11、PyTorch 2.9.1、CUDA 12.8；其他 GPU/驱动按
[官方版本表](https://pytorch.org/get-started/previous-versions/)选择兼容构建。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[prepare]'
python -m pip check
mkdir -p data weights runs
```

几何推理需另行[安装授权的 VGGT-Omega](docs/DEPENDENCIES.md#installation)，准备 `weights/omega.pt`。
GPU 不支持 BF16 时，删掉后续命令中的 `--bf16`。

## 2. 准备数据

将训练视频放入 `data/raw/train`。先准备冻结编码器的源码和权重，修改
`configs/prepare_wan.json` 或 `configs/prepare_vrae.json` 中的路径，具体见[编码器配置](docs/PREPARATION.md#install-encoders)。

```bash
python prepare_data.py --config configs/prepare_wan.json \
  --videos data/raw/train --output data/wan_train --device cuda --bf16

python prepare_data.py --config configs/prepare_vrae.json \
  --videos data/raw/train --output data/vrae_train --device cuda --bf16
```

默认每个视频取前 49 帧，中心裁剪为方形后缩放到 560x560。**先按序列划分训练/测试集**再生成缓存。
短视频或损坏视频会报错，不补假帧、不静默跳过。其他起始帧或多窗口用[视频清单](docs/PREPARATION.md#explicit-windows)指定。

输出 `pairs/*.pt`、溯源记录、`train.jsonl` 和 `COMPLETE.json`。
每个 pair 包含不带 batch 维的 `latent` 与 `teacher_tokens`：

| 路线 | 训练 latent | teacher tokens |
|---|---|---|
| WAN | `[16,13,70,70]`，VAE 原生缩放 | `[49,1225,1024]` |
| V-RAE | `[13,1024,35,35]`，未施加 diffusion normalizer 的原生特征 | `[49,1225,1024]` |

teacher 使用同一段 49 帧和裁剪；只有 V-RAE 在编码前额外补 3 个首帧并池化。
仅 teacher 缓存每个窗口就约占 117 MiB（BF16）或 234 MiB（FP32），需预留磁盘。
更多格式、权重依赖和失败处理见[数据准备说明](docs/PREPARATION.md)。

## 3. 训练

以下为示例配方，不是历史多阶段训练的精确复现。两条路线分别运行：

```bash
python train.py --config configs/wan.json --manifest data/wan_train/train.jsonl \
  --output runs/wan_train --steps 94000 --batch-size 1 --lr 2e-4 \
  --save-every 1000 --device cuda --bf16

python train.py --config configs/vrae.json --manifest data/vrae_train/train.jsonl \
  --output runs/vrae_train --steps 50000 --batch-size 1 --lr 5e-5 \
  --save-every 1000 --device cuda --bf16
```

首次可改成 `--steps 2 --save-every 1`，用新目录先跑通两步。
成功后保存 `step_*.pt` 和 `TRAINING_COMPLETE.json`。

单机 8 卡：

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/wan.json --manifest data/wan_train/train.jsonl \
  --output runs/wan_train_8gpu --steps 94000 --batch-size 1 --lr 2e-4 \
  --save-every 1000 --device cuda --bf16
```

V-RAE 将配置、清单、输出目录、步数和学习率替换为上面对应值。
batch 为每卡 batch，上例全局 batch 为 8。只训练 Bridge。
输出目录必须不存在；当前不支持 optimizer 续训或自动验证/最优权重选择。

## 4. 推理

使用训练得到的 Bridge 权重和准备好的原生 latent：

```bash
python infer.py --checkpoint runs/wan_train/step_094000.pt \
  --omega-checkpoint weights/omega.pt --latent data/wan_latent.pt \
  --output runs/wan_infer --device cuda --bf16 --export-ply
```

V-RAE 改用 `runs/vrae_train/step_050000.pt`、`data/vrae_latent.pt` 和新输出目录。
需要点云颜色时加 `--rgb-frames data/rgb_uint8.npy`，RGB 为 uint8
`[49,560,560,3]`，帧序/裁剪与 latent 一致。

输出 `depth.npy`、`cameras.npz`、49 个全密度 PLY 和 `COMPLETE.json`。
只需要深度/相机时删掉 `--export-ply`。
完整 49 帧一起解码，不隐式使用 GT 位姿或深度过滤。

## 5. 指标评估

按[评估输入说明](docs/EVALUATION.md#prepare-an-evaluation-bundle)将预测深度/相机与对应 GT
打包成 NPZ，再计算指标：

```bash
python evaluate.py --bundle data/wan_eval_case.npz --output runs/wan_eval \
  --filter log_mad_multiview --camera omega --sampling all --unit native
```

无过滤对照改为 `--filter none` 并使用新输出目录；V-RAE 换成对应的评估 NPZ。
过滤复用已有深度，不重新生成。结果为 `metrics.json` 和 `prediction_mask.npz`；
Accuracy、Completeness、Chamfer 都是越低越好的距离。
只有独立确认 GT 为米制才使用 `--unit meters`，CO3D/WildRGB-D 分开统计。

更多参数：`python train.py --help`、`python infer.py --help`、`python evaluate.py --help`。
完整协议见[评估文档](docs/EVALUATION.md)。
