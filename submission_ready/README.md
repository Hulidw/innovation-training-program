# 基于动态参数分配的 LoRA 大语言模型高效微调优化方法

本项目实现赛题一的动态 LoRA 方法：在固定总 rank 预算下，根据训练期间的层重要性为不同 Transformer 层动态分配 LoRA rank。

## 核心方法

1. 用 LoRA 梯度与更新幅度的交叉乘积估计层重要性。
2. 用指数滑动平均降低小批次噪声。
3. 在 `sum(rank)=预算`、`min_rank<=rank<=max_rank` 的约束下按层调整 rank。
4. 预分配最大 rank 张量，前向时切片，调整 rank 时不重新初始化参数。
5. Dynamic-U 可额外使用重要性信号的 LCB 置信下界，降低不可靠信号引起的 rank 抖动。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 本地快速演示

```powershell
python simple_train.py --steps 100 --output runs/simple
python train.py --mode compare --steps 300 --output runs/compare
```

结果包含损失、速度、峰值显存、预分配参数量、当前有效 rank 参数量和 rank 历史。

## 真实模型实验

`real_model_experiment.py` 使用 Qwen2.5-0.5B-Instruct 和 GLUE SST-2，支持 Uniform LoRA、Dynamic LoRA、Dynamic-U 和 AdaLoRA。

```powershell
python real_model_experiment.py --modes uniform dynamic adaloora --steps 1500 --train-samples 3000 --eval-samples 200 --device cuda --output runs/real_model
```

运行完整消融对比：

```powershell
python real_model_experiment.py --modes uniform dynamic dynamic_u adaloora --uncertainty-beta 1.0 --steps 1500 --device cuda --output runs/four_way
```

如果模型和数据集已经缓存，可以设置 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1`，避免重复联网检查。

## 云端 Qwen 实验

```powershell
python cloud_qwen_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 60 --device cuda --output runs/qwen25_05b
```

详细步骤见 `云端Qwen实验说明.md`。

## 文件说明

- `dynamic_lora.py`：动态 LoRA 层和 rank 分配器。
- `real_model_experiment.py`：Qwen + SST-2 真实模型实验。
- `adaloora.py`：AdaLoRA 对比基线。
- `cloud_qwen_experiment.py`：云端轻量 Qwen 对比实验。
- `generate_report.py`：汇总实验结果。
- `generate_submission.py`：同步提交目录和技术方案材料。
