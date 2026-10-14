"""真实模型实验：Qwen2.5-0.5B-Instruct + GLUE SST-2 情感分类。

对比 Uniform LoRA 与 Dynamic LoRA 在真实大模型微调任务上的表现。
评估指标：验证集准确率、验证 loss、显存峰值、可训练参数量、训练吞吐量。
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from dynamic_lora import (
    AllocationConfig,
    DynamicRankAllocator,
    inject_dynamic_lora,
)
from adaloora import AdaLoRAAllocator, inject_adaloora


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_prompt(sentence: str) -> str:
    """将 SST-2 句子转为指令格式。"""
    return (
        f"Review: {sentence}\n"
        f"Sentiment (positive/negative):"
    )


LABEL_TO_WORD = {0: "negative", 1: "positive"}
WORD_TO_LABEL = {"negative": 0, "positive": 1}


def build_chat(tokenizer, prompt: str, answer: str) -> str:
    """用 chat template 拼接指令与回答。"""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"User: {prompt}\nAssistant: {answer}"


def make_eval_prompt(tokenizer, prompt: str) -> str:
    """评估时仅拼接用户侧，留给模型生成。"""
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {prompt}\nAssistant:"


def collate(tokenizer, texts, device, max_length):
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


@torch.no_grad()
def evaluate(model, tokenizer, eval_texts, eval_labels, device, max_length, batch_size=16):
    """用 generation 评估分类准确率。"""
    model.eval()
    correct = 0
    total = 0
    losses = []
    for i in range(0, len(eval_texts), batch_size):
        prompts = eval_texts[i : i + batch_size]
        labels = eval_labels[i : i + batch_size]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # 计算 eval loss（用完整的 prompt+answer）
        full_texts = [p + " " + LABEL_TO_WORD[l] for p, l in zip(prompts, labels)]
        full = collate(tokenizer, full_texts, device, max_length)
        with torch.no_grad():
            loss = model(**full).loss
            losses.append(float(loss.item()))
        # 生成预测
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for gen, label in zip(generated, labels):
            gen_lower = gen.strip().lower()
            pred = WORD_TO_LABEL.get(gen_lower[:8], -1)
            if pred == label:
                correct += 1
            total += 1
    model.train()
    return correct / total if total else 0.0, sum(losses) / len(losses) if losses else float("inf")


def run_experiment(args, mode: str, tokenizer, train_texts, train_labels, eval_texts, eval_labels, device):
    """运行单组实验（uniform / dynamic / adaloora）。"""
    seed_all(args.seed)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

    if mode == "adaloora":
        # AdaLoRA: SVD 形式 + 全局奇异值预算
        layers = inject_adaloora(
            model,
            target_names=tuple(args.target_names),
            max_rank=args.max_rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
        if not layers:
            raise RuntimeError(f"未找到目标层: {args.target_names}")
        for p in model.parameters():
            p.requires_grad_(False)
        for layer in layers.values():
            layer.P.requires_grad_(True)
            layer.Lambda.requires_grad_(True)
            layer.Q.requires_grad_(True)
        allocator = AdaLoRAAllocator(layers, budget=args.rank_budget)
        allocator.initialize_uniform()
    else:
        # Uniform / Dynamic: BA 形式 + 层级 rank 分配
        layers = inject_dynamic_lora(
            model,
            target_names=tuple(args.target_names),
            max_rank=args.max_rank,
            rank=args.initial_rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
        if not layers:
            raise RuntimeError(f"未找到目标层: {args.target_names}")
        for p in model.parameters():
            p.requires_grad_(False)
        for layer in layers.values():
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)
        allocator = DynamicRankAllocator(
            layers,
            AllocationConfig(
                budget=args.rank_budget,
                min_rank=args.min_rank,
                max_rank=args.max_rank,
                step=args.rank_step,
                switch_threshold=args.switch_threshold,
                ema_decay=args.ema_decay,
                uncertainty_beta=args.uncertainty_beta if mode == "dynamic_u" else 0.0,
                min_samples=args.min_samples,
            ),
        )
        allocator.initialize_uniform()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    losses = []
    start = time.perf_counter()
    steps_per_epoch = (len(train_texts) + args.batch_size - 1) // args.batch_size

    for step in range(1, args.steps + 1):
        idx = (step - 1) % steps_per_epoch
        batch_texts = train_texts[idx * args.batch_size : (idx + 1) * args.batch_size]
        batch_labels = train_labels[idx * args.batch_size : (idx + 1) * args.batch_size]
        full = [p + " " + LABEL_TO_WORD[l] for p, l in zip(batch_texts, batch_labels)]
        batch = collate(tokenizer, full, device, args.max_length)

        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        if mode in ("dynamic", "dynamic_u", "adaloora"):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
            )
            if mode in ("dynamic", "dynamic_u"):
                allocator.observe()
        optimizer.step()

        if mode in ("dynamic", "dynamic_u") and step >= args.warmup_steps and step % args.allocation_interval == 0:
            allocator.rebalance(step, update_importance=False)
        if mode == "adaloora" and step >= args.warmup_steps and step % args.allocation_interval == 0:
            allocator.reallocate(step)

        losses.append(float(loss.detach().float().cpu()))

    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    accuracy, eval_loss = evaluate(
        model, tokenizer, eval_texts, eval_labels, device, args.max_length, args.eval_batch_size
    )

    if mode == "adaloora":
        active_adapter_params = sum(
            int(layer.active_mask.sum().item()) * (layer.base.in_features + layer.base.out_features)
            for layer in layers.values()
        )
    else:
        active_adapter_params = sum(layer.active_parameter_count for layer in layers.values())

    result = {
        "mode": mode,
        "model": args.model,
        "device": str(device),
        "steps": args.steps,
        "train_size": len(train_texts),
        "eval_size": len(eval_texts),
        "target_layers": len(layers),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "active_adapter_params": active_adapter_params,
        "uncertainty_beta": args.uncertainty_beta if mode == "dynamic_u" else 0.0,
        "final_train_loss": losses[-1],
        "mean_last_20_train_loss": sum(losses[-20:]) / min(20, len(losses)),
        "eval_loss": eval_loss,
        "eval_accuracy": accuracy,
        "peak_memory_gb": peak_mem,
        "seconds": elapsed,
        "steps_per_sec": args.steps / elapsed,
        "rank_history": allocator.history if mode in ("dynamic", "dynamic_u", "adaloora") else [],
    }

    del optimizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_and_prepare_data(args):
    """加载 GLUE SST-2 数据集并预处理为指令格式。"""
    print(f"[INFO] 加载数据集 GLUE SST-2 ...")
    dataset = load_dataset("glue", "sst2")
    train_data = list(dataset["train"])
    val_data = list(dataset["validation"])

    random.seed(args.seed)
    random.shuffle(train_data)
    if args.train_samples > 0:
        train_data = train_data[: args.train_samples]
    if args.eval_samples > 0:
        val_data = val_data[: args.eval_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts = [build_chat(tokenizer, make_prompt(d["sentence"]), LABEL_TO_WORD[d["label"]]) for d in train_data]
    train_labels = [d["label"] for d in train_data]
    eval_prompts = [make_eval_prompt(tokenizer, make_prompt(d["sentence"])) for d in val_data]
    eval_labels = [d["label"] for d in val_data]
    return train_texts, train_labels, eval_prompts, eval_labels, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=3000, help="训练样本数，0 表示全部")
    parser.add_argument("--eval-samples", type=int, default=200, help="评估样本数，0 表示全部")
    parser.add_argument("--rank-budget", type=int, default=64)
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--initial-rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank-step", type=int, default=1)
    parser.add_argument("--allocation-interval", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--switch-threshold", type=float, default=0.3)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0,
                        help="dynamic_u 模式的 LCB 置信系数 beta（0 关闭）")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="应用 LCB 前的最少重要性观测步数")
    parser.add_argument("--modes", nargs="+",
                        default=["uniform", "dynamic", "dynamic_u", "adaloora"],
                        choices=["uniform", "dynamic", "dynamic_u", "adaloora"],
                        help="要运行的对比方法，按给定顺序执行并输出")
    parser.add_argument("--output", default="runs/real_model")
    parser.add_argument("--target-names", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.max_length < 8:
        raise SystemExit("steps, batch-size and max-length must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    train_texts, train_labels, eval_prompts, eval_labels, tokenizer = load_and_prepare_data(args)
    print(f"[INFO] 训练样本: {len(train_texts)}, 评估样本: {len(eval_prompts)}")
    print(f"[INFO] 目标层: {args.target_names}, rank_budget={args.rank_budget}, max_rank={args.max_rank}")
    if "dynamic_u" in args.modes and args.uncertainty_beta < 0:
        raise SystemExit("uncertainty-beta must be non-negative")

    results = []
    for mode in args.modes:
        print(f"\n{'='*60}")
        print(f"[INFO] 开始 {mode} 实验 ...")
        print(f"{'='*60}")
        r = run_experiment(args, mode, tokenizer, train_texts, train_labels, eval_prompts, eval_labels, device)
        results.append(r)
        print(f"[RESULT] {mode}: acc={r['eval_accuracy']:.4f}, eval_loss={r['eval_loss']:.4f}, "
              f"train_loss={r['final_train_loss']:.4f}, peak_mem={r['peak_memory_gb']:.2f}GB, "
              f"params={r['trainable_params']}, time={r['seconds']:.1f}s")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "real_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 按运行顺序生成对比摘要
    labels = {
        "uniform": "Uniform LoRA",
        "dynamic": "Dynamic (EMA)",
        "dynamic_u": "Dynamic-U (ours)",
        "adaloora": "AdaLoRA",
    }
    modes = [r["mode"] for r in results]
    metrics = [
        ("验证准确率", "eval_accuracy", ".4f"),
        ("验证损失", "eval_loss", ".4f"),
        ("最终训练损失", "final_train_loss", ".4f"),
        ("峰值显存(GB)", "peak_memory_gb", ".2f"),
        ("可训练参数", "trainable_params", ""),
        ("训练时间(s)", "seconds", ".1f"),
        ("吞吐量(steps/s)", "steps_per_sec", ".2f"),
    ]
    lines = [
        "# 真实模型实验结果（Qwen2.5-0.5B-Instruct + GLUE SST-2）\n",
        "| 指标 | " + " | ".join(labels[m] for m in modes) + " |",
        "|---" * (len(modes) + 1) + "|",
    ]
    for name, key, fmt in metrics:
        cells = []
        for r in results:
            v = r[key]
            cells.append(f"{v:{fmt}}" if fmt else str(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"训练样本: {len(train_texts)}, 评估样本: {len(eval_prompts)}, 步数: {args.steps}, "
        f"rank_budget={args.rank_budget}, uncertainty_beta={args.uncertainty_beta}\n"
    )
    summary = "\n".join(lines)
    (output / "real_results.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"[INFO] 结果已保存到 {output}")


if __name__ == "__main__":
    main()
