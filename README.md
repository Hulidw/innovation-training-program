# 基于动态参数分配的 LoRA 大语言模型高效微调优化方法

本项目实现赛题 1 的一个可复现实验版本：在 LoRA 的低秩更新基础上，根据每个目标层的梯度敏感性和 LoRA 更新幅度估计重要性，在固定总 rank 预算下动态调整各层 rank。

## 核心方法

1. 用 `importance = ema(|grad_A|*|B| + |grad_B|*|A|)` 估计层重要性，并在每个训练步累计指数滑动平均。
2. 每隔 `allocation_interval` 个优化步，根据平滑后的重要性将总 rank 预算小步分配；rank 在 `[min_rank, max_rank]` 内按步长增减。
3. 采用可扩展的最大 rank 参数矩阵，当前 rank 通过切片参与前向，因此重分配不会破坏已有参数。
4. 固定总 rank 后，对比均匀 LoRA 和动态 LoRA 的验证损失、可训练参数量、显存和训练耗时。

当前实现还加入了两项稳定性处理：rank 从当前分配小步调整，避免周期性重置；LoRA 更新使用固定的 `alpha / max_rank` 缩放，避免 rank 调整时已学到的更新突然改变。
均匀 LoRA 和动态 LoRA 都从同一个预算平衡的初始 rank 开始，动态方案默认使用较保守的切换阈值和 EMA 衰减，减少小批次噪声带来的 rank 抖动。

## 快速开始

如果只想快速演示算法，不需要配置参数，直接运行：

```powershell
python simple_train.py --steps 100
```

该脚本只使用一个小型前馈网络，展示“计算层重要性 → 在固定预算内重新分配 rank”的完整流程。

```powershell
cd dynamic_lora_project
python -m pip install -r requirements.txt
python train.py --model tiny --steps 300 --output runs/tiny
```

没有 GPU 也可以运行；有 CUDA 时自动使用 CUDA。`tiny` 是本地可运行的字符级语言模型演示，不需要下载模型或数据。接入 Hugging Face 模型时，保留 `DynamicLoRA` 和 `DynamicRankAllocator`，将目标线性层替换为 `DynamicLoRALinear` 即可。

```powershell
python train.py --model tiny --mode compare --steps 300 --output runs/compare
```

需要调整动态分配稳定性时，可使用 `--switch-threshold` 和 `--ema-decay`；例如：

```powershell
python train.py --mode dynamic --steps 300 --switch-threshold 0.5 --ema-decay 0.95 --output runs/dynamic
```

结果包括 `metrics.json`（含 rank 分配历史）、`losses.json` 和 `checkpoint.pt`。

## Qwen 云端实验

使用 `cloud_qwen_experiment.py` 可以在 CUDA 云端对 `Qwen/Qwen2.5-0.5B-Instruct` 做固定 rank 与动态 rank 对比：

```powershell
python cloud_qwen_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 60 --device cuda --output runs/qwen25_05b
```

详细步骤见 `云端Qwen实验说明.md`。
# ʵ���¼ 
