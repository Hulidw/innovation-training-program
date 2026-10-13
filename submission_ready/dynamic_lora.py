"""Dynamic-rank LoRA components.

The implementation keeps max-rank tensors and activates a prefix according to
the current allocation, so rank changes are cheap and optimizer state remains
valid.
"""
from dataclasses import dataclass
import math
from typing import Dict, List
import torch
from torch import nn
import torch.nn.functional as F


class DynamicLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, max_rank: int = 8, rank: int = 2,
                 alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        if rank < 1 or rank > max_rank:
            raise ValueError("rank must be between 1 and max_rank")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.max_rank, self.rank, self.alpha = max_rank, rank, float(alpha)
        self.scaling = self.alpha / max_rank
        self.dropout = nn.Dropout(dropout)
        # Wrapping an already CUDA/half-precision layer must preserve its placement.
        factory = {'device': base.weight.device, 'dtype': base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(max_rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, max_rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.register_buffer("importance_ema", torch.zeros((), device=base.weight.device))

    def set_rank(self, rank: int):
        if not 1 <= rank <= self.max_rank:
            raise ValueError('rank must be between 1 and max_rank')
        self.rank = int(rank)

    def get_extra_state(self):
        # Plain Python attributes otherwise disappear in a state_dict checkpoint.
        return {'rank': self.rank, 'alpha': self.alpha}

    def set_extra_state(self, state):
        self.set_rank(state['rank'])
        self.alpha = float(state['alpha'])
        self.scaling = self.alpha / self.max_rank

    def forward(self, x):
        y = self.base(x)
        r = self.rank
        update = F.linear(F.linear(self.dropout(x), self.lora_A[:r]), self.lora_B[:, :r])
        # Scale by alpha / max_rank (constant). Scaling by the active rank
        # would rescale the already-learned update whenever the allocator
        # changes r, causing an abrupt change of the network function.
        return y + update * self.scaling

    @torch.no_grad()
    def update_importance(self, decay: float = 0.9):
        if self.lora_A.grad is None or self.lora_B.grad is None:
            return self.importance_ema
        r = self.rank
        # First-order sensitivity of the active low-rank update.
        score = (self.lora_A.grad[:r].float().abs().mean() * self.lora_B[:, :r].float().abs().mean()
                 + self.lora_A[:r].float().abs().mean() * self.lora_B.grad[:, :r].float().abs().mean())
        self.importance_ema.mul_(decay).add_(score * (1 - decay))
        return self.importance_ema


@dataclass
class AllocationConfig:
    budget: int = 16
    min_rank: int = 1
    max_rank: int = 8
    step: int = 1
    # Conservative defaults reduce rank churn caused by minibatch noise.
    switch_threshold: float = 0.5
    ema_decay: float = 0.95


class DynamicRankAllocator:
    def __init__(self, layers: Dict[str, DynamicLoRALinear], config: AllocationConfig):
        if not layers:
            raise ValueError('no LoRA layers found')
        if any(type(v) is not int for v in (config.budget, config.min_rank, config.max_rank, config.step)):
            raise ValueError('rank budget, bounds and step must be integers')
        if config.budget < 1 or config.min_rank < 1 or config.max_rank < config.min_rank or config.step < 1:
            raise ValueError('invalid rank budget, bounds or step')
        if any(layer.max_rank < config.max_rank for layer in layers.values()):
            raise ValueError('allocator max_rank exceeds a layer capacity')
        if not 0 <= config.ema_decay < 1 or not math.isfinite(config.switch_threshold) or config.switch_threshold < 0:
            raise ValueError('invalid EMA decay or switching threshold')
        self.layers, self.config = layers, config
        self.effective_budget = max(config.min_rank * len(layers), min(config.budget, config.max_rank * len(layers)))
        self.history: List[dict] = []

    def initialize_uniform(self):
        """The two comparison methods start with exactly the same active budget."""
        base, remainder = divmod(self.effective_budget, len(self.layers))
        for i, layer in enumerate(self.layers.values()):
            layer.set_rank(base + (i < remainder))
        return {n: layer.rank for n, layer in self.layers.items()}

    @torch.no_grad()
    def observe(self):
        """Call after backward and before optimizer.step(), on every training step."""
        for layer in self.layers.values():
            layer.update_importance(self.config.ema_decay)

    def _project(self, scores: Dict[str, float]) -> Dict[str, int]:
        names = list(self.layers)
        c = self.config
        budget = self.effective_budget
        if any(not math.isfinite(v) or v < 0 for v in scores.values()):
            raise ValueError('importance scores must be finite and non-negative')
        # Start from the current allocation. Small changes preserve optimizer
        # stability and avoid dropping a layer from a useful rank to min_rank.
        ranks = {n: max(c.min_rank, min(c.max_rank, int(self.layers[n].rank))) for n in names}
        while sum(ranks.values()) > budget:
            candidates = [n for n in names if ranks[n] > c.min_rank]
            n = min(candidates, key=lambda x: scores.get(x, 0.0))
            ranks[n] -= min(c.step, ranks[n] - c.min_rank, sum(ranks.values()) - budget)
        while sum(ranks.values()) < budget:
            candidates = [n for n in names if ranks[n] < c.max_rank]
            n = max(candidates, key=lambda x: scores.get(x, 0.0))
            ranks[n] += min(c.step, c.max_rank - ranks[n], budget - sum(ranks.values()))
        # When the budget is already full, make at most one stable swap per
        # rebalance instead of repeatedly concentrating all capacity in one layer.
        # Skip the swap unless the importance gap is significant: reallocating
        # on noise-level differences churns ranks without benefit.
        donors = [n for n in names if ranks[n] - c.step >= c.min_rank]
        receivers = [n for n in names if ranks[n] + c.step <= c.max_rank]
        if donors and receivers:
            donor = min(donors, key=lambda x: scores.get(x, 0.0))
            receiver = max(receivers, key=lambda x: scores.get(x, 0.0))
            if scores.get(receiver, 0.0) > scores.get(donor, 0.0) * (1 + c.switch_threshold):
                ranks[donor] -= c.step
                ranks[receiver] += c.step
        return ranks

    @torch.no_grad()
    def rebalance(self, step: int, update_importance: bool = True):
        if update_importance:
            self.observe()
        # One device-to-host transfer at allocation time, not one per layer/step.
        values = torch.stack([l.importance_ema for l in self.layers.values()]).cpu().tolist()
        scores = dict(zip(self.layers, values))
        ranks = self._project(scores)
        for n, r in ranks.items(): self.layers[n].set_rank(r)
        self.history.append({"step": step, "ranks": ranks, "importance": scores})
        return ranks


def inject_dynamic_lora(model: nn.Module, target_names=("q_proj", "v_proj", "linear"),
                        max_rank=8, rank=2, alpha=16.0, dropout=0.05):
    """Replace matching nn.Linear children recursively and return a name->layer map."""
    found = {}
    def visit(parent, prefix=""):
        for name, child in list(parent.named_children()):
            if isinstance(child, DynamicLoRALinear):
                continue
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(k in name or k in full for k in target_names):
                wrapped = DynamicLoRALinear(child, max_rank, rank, alpha, dropout)
                setattr(parent, name, wrapped); found[full] = wrapped
            else: visit(child, full)
    visit(model)
    return found
