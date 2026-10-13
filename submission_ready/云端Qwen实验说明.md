# Qwen2.5-0.5B-Instruct 云端实验

本实验选择 \`Qwen/Qwen2.5-0.5B-Instruct\`，原因是模型较小、下载和显存要求低，适合先验证动态 LoRA。实验比较固定 rank LoRA 和动态 rank LoRA，二者使用相同模型、随机种子、数据、训练步数和总 rank 预算。

## 云端运行

在 Colab、AutoDL 或其他 CUDA 云端环境中进入项目目录，先安装依赖：

\`\`\`bash
pip install -r requirements.txt
\`\`\`

然后运行：

\`\`\`bash
python cloud_qwen_experiment.py \\
  --model Qwen/Qwen2.5-0.5B-Instruct \\
  --steps 60 \\
  --device cuda \\
  --output runs/qwen25_05b
\`\`\`

程序会自动下载模型，分别运行固定 rank 和动态 rank，并将结果保存到：

\`\`\`text
runs/qwen25_05b/qwen_results.json
\`\`\`

结果重点查看 \`eval_loss\`、\`mean_last_20\`、\`seconds\`、\`steps_per_sec\` 和 \`rank_history\`。如果动态方案的验证损失更低或速度相近，就保留当前参数；如果 rank 变化过快，可提高 \`--switch-threshold\` 或 \`--ema-decay\`。

## 显存不足时

先减少序列长度和 batch：

\`\`\`bash
python cloud_qwen_experiment.py --steps 30 --batch-size 1 --max-length 128 --device cuda
\`\`\`

正式比赛材料中应使用云端生成的 JSON 和曲线，不能把本地小测试结果当作正式成绩。
