"""最简动态 LoRA 演示：固定总 rank，按层梯度重要性动态分配。"""
import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from dynamic_lora import DynamicLoRALinear, DynamicRankAllocator, AllocationConfig

class Model(nn.Module):
    def __init__(self, d=64, max_rank=8):
        super().__init__()
        self.layers = nn.ModuleList([
            DynamicLoRALinear(nn.Linear(d, d), max_rank=max_rank, rank=2)
            for _ in range(4)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--budget', type=int, default=12)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--output', default='runs/simple')
    a = p.parse_args()
    if a.steps < 1 or a.budget < 1 or a.interval < 1:
        raise SystemExit('steps, budget and interval must be positive')
    random.seed(42); torch.manual_seed(42)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    m = Model()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.01)
    layers = {f'layers.{i}': layer for i, layer in enumerate(m.layers)}
    allocator = DynamicRankAllocator(layers, AllocationConfig(budget=a.budget, min_rank=1, max_rank=8, step=1, switch_threshold=0.5, ema_decay=0.95))
    allocator.initialize_uniform()
    for step in range(1, a.steps + 1):
        x = torch.randn(32, 64); target = torch.zeros_like(x)
        loss = ((m(x) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); allocator.observe(); opt.step()
        if step % a.interval == 0:
            allocator.rebalance(step, update_importance=False)
    json.dump(allocator.history, open(out / 'rank_history.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    torch.save(m.state_dict(), out / 'model.pt')
    print('完成，结果保存到', out)

if __name__ == '__main__':
    main()
