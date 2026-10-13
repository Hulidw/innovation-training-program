"""生成提交材料需要的最小可复现实验：均匀 LoRA vs 动态 LoRA。"""
import json, time
from pathlib import Path
import torch
from simple_train import Model
from dynamic_lora import DynamicRankAllocator, AllocationConfig

def run(mode, steps=100, budget=12, interval=10):
    torch.manual_seed(42)
    model = Model()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
    layers = {f'layers.{i}': layer for i, layer in enumerate(model.layers)}
    allocator = DynamicRankAllocator(layers, AllocationConfig(budget=budget, min_rank=1, max_rank=8, step=1, switch_threshold=0.5, ema_decay=0.95))
    allocator.initialize_uniform()
    losses = []
    t = time.time()
    for step in range(1, steps + 1):
        x = torch.randn(32, 64); target = torch.zeros_like(x)
        loss = ((model(x) - target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        if mode == 'dynamic':
            allocator.observe()
        opt.step()
        losses.append(float(loss.item()))
        if mode == 'dynamic' and step % interval == 0:
            allocator.rebalance(step, update_importance=False)
    if not losses:
        raise ValueError('steps must be positive')
    recent = losses[-20:]
    return {'mode': mode, 'steps': steps, 'budget': budget,
            'final_loss': losses[-1], 'mean_last_20': sum(recent) / len(recent),
            'seconds': time.time() - t, 'ranks': allocator.history}

if __name__ == '__main__':
    out = Path('runs/submission_experiment'); out.mkdir(parents=True, exist_ok=True)
    result = [run('uniform'), run('dynamic')]
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'results.md').write_text('# 实验结果\n\n' + ''.join(
        f"- {x['mode']}: final_loss={x['final_loss']:.6f}, mean_last_20={x['mean_last_20']:.6f}, seconds={x['seconds']:.2f}\n"
        for x in result), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
