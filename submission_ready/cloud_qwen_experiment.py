"""Qwen2.5-0.5B-Instruct cloud comparison for fixed and dynamic LoRA."""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dynamic_lora import AllocationConfig, DynamicRankAllocator, inject_dynamic_lora


EXAMPLES = [
    ("把下面的句子翻译成英文：今天天气很好。", "The weather is very nice today."),
    ("把下面的句子翻译成英文：我喜欢学习人工智能。", "I like learning artificial intelligence."),
    ("计算 12 加 2 等于多少？只输出结果。", "14"),
    ("计算 9 乘 7 等于多少？只输出结果。", "63"),
    ("用一句话解释什么是机器学习。", "Machine learning is a method that lets computers learn patterns from data."),
    ("用一句话解释什么是 LoRA。", "LoRA is a parameter efficient method that learns low rank updates."),
    ("将这句话改得更正式：这个方案挺好用。", "这个方案具有较好的实用性。"),
    ("将这句话改得更简洁：我们将在明天对模型进行实验测试。", "我们明天测试模型。"),
    ("列出两个深度学习框架。", "PyTorch 和 TensorFlow。"),
    ("列出两个常见的编程语言。", "Python 和 Java。"),
    ("回答是或否：Python 是一种编程语言。", "是。"),
    ("回答是或否：CUDA 是一种数据库。", "否。"),
]


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_text(tokenizer, prompt: str, answer: str) -> str:
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"User: {prompt}\nAssistant: {answer}"


def make_batch(tokenizer, texts, device, max_length):
    batch = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][batch["attention_mask"] == 0] = -100
    return batch


def run_once(args, mode, tokenizer, texts, device):
    seed_all(args.seed)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    layers = inject_dynamic_lora(
        model,
        target_names=("q_proj", "v_proj"),
        max_rank=args.max_rank,
        rank=args.initial_rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    if not layers:
        raise RuntimeError("没有找到 Qwen 的 q_proj/v_proj 目标层")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in layers.values():
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
    allocator = DynamicRankAllocator(
        layers,
        AllocationConfig(
            budget=args.rank_budget,
            min_rank=1,
            max_rank=args.max_rank,
            step=1,
            switch_threshold=args.switch_threshold,
            ema_decay=args.ema_decay,
        ),
    )
    allocator.initialize_uniform()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    model.train()
    losses = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch_texts = [texts[(step - 1 + i) % len(texts)] for i in range(args.batch_size)]
        batch = make_batch(tokenizer, batch_texts, device, args.max_length)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        if mode == "dynamic":
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
            )
            allocator.observe()
        optimizer.step()
        if mode == "dynamic" and step >= args.warmup_steps and step % args.allocation_interval == 0:
            allocator.rebalance(step, update_importance=False)
        losses.append(float(loss.detach().float().cpu()))
    elapsed = time.perf_counter() - start
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    active_adapter_params = sum(layer.active_parameter_count for layer in layers.values())
    model.eval()
    with torch.no_grad():
        eval_batch = make_batch(tokenizer, texts[: args.batch_size], device, args.max_length)
        eval_loss = float(model(**eval_batch).loss.float().cpu())
    result = {
        "mode": mode,
        "model": args.model,
        "device": str(device),
        "steps": args.steps,
        "train_loss": losses[-1],
        "mean_last_20": sum(losses[-20:]) / min(20, len(losses)),
        "eval_loss": eval_loss,
        "seconds": elapsed,
        "steps_per_sec": args.steps / elapsed,
        "target_layers": len(layers),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "active_adapter_params": active_adapter_params,
        "peak_memory_gb": peak_memory_gb,
        "rank_history": allocator.history,
    }
    del optimizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rank-budget", type=int, default=96)
    parser.add_argument("--max-rank", type=int, default=8)
    parser.add_argument("--initial-rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--allocation-interval", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--switch-threshold", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/qwen25_05b")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.max_length < 8:
        raise SystemExit("steps, batch-size and max-length must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [make_text(tokenizer, p, a) for p, a in EXAMPLES]
    results = [
        run_once(args, "uniform", tokenizer, texts, device),
        run_once(args, "dynamic", tokenizer, texts, device),
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "qwen_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
