"""多预算对比实验：在不同 rank budget 下对比 Uniform vs Dynamic LoRA。"""
import argparse
import json
from pathlib import Path

from real_model_experiment import run_experiment, load_and_prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--budgets", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--initial-rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank-step", type=int, default=1)
    parser.add_argument("--allocation-interval", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--switch-threshold", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/budget_sweep")
    parser.add_argument("--target-names", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    train_texts, train_labels, eval_prompts, eval_labels, tokenizer = load_and_prepare_data(args)

    all_results = []
    for budget in args.budgets:
        args.rank_budget = budget
        print(f"\n{'#'*70}")
        print(f"# Budget = {budget}")
        print(f"{'#'*70}")
        for mode in ("uniform", "dynamic"):
            print(f"\n--- {mode} LoRA (budget={budget}) ---")
            r = run_experiment(args, mode, tokenizer, train_texts, train_labels, eval_prompts, eval_labels, device)
            r["budget"] = budget
            all_results.append(r)
            print(f"  acc={r['eval_accuracy']:.4f}, eval_loss={r['eval_loss']:.4f}, "
                  f"train_loss={r['final_train_loss']:.4f}, mem={r['peak_memory_gb']:.2f}GB")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "budget_sweep.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成对比表格
    lines = ["# 多预算对比实验结果（Qwen2.5-0.5B-Instruct + GLUE SST-2）\n"]
    lines.append("| Budget | 模式 | 准确率 | 验证损失 | 训练损失 | 显存(GB) | 参数量 | 时间(s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in all_results:
        lines.append(
            f"| {r['budget']} | {r['mode']} | {r['eval_accuracy']:.4f} | {r['eval_loss']:.4f} | "
            f"{r['final_train_loss']:.4f} | {r['peak_memory_gb']:.2f} | {r['trainable_params']} | {r['seconds']:.1f} |"
        )
    lines.append("")
    (output / "budget_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n[INFO] 结果保存到 {output}")


if __name__ == "__main__":
    main()
