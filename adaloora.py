"""AdaLoRA 基线实现。

与 DynamicLoRA 的关键区别：
1. 参数化：SVD 形式 ΔW = P·diag(Λ)·Qᵀ，而非 BA 形式
2. 重要性粒度：奇异值级别（而非层级别），I_i = |∂L/∂Λ_i|·|Λ_i|
3. 预算分配：跨所有层的全局奇异值预算，保留重要性最高的 budget 个奇异值
4. 无 EMA 平滑，使用单步梯度敏感度

参考: Zhang et al., AdaLoRA: Adaptive Budget Allocation for PEFT, NeurIPS 2023
"""
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F


class AdaLoRALinear(nn.Module):
    """SVD 参数化的 LoRA 层。ΔW = P · diag(Λ) · Qᵀ"""

    def __init__(self, base: nn.Linear, max_rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.max_rank = max_rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / max_rank
        self.dropout = nn.Dropout(dropout)

        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        in_f, out_f = base.in_features, base.out_features
        # SVD 分解: P (out, r), Λ (r,), Q (r, in)
        self.P = nn.Parameter(torch.empty(out_f, max_rank, **factory))
        # Λ 初始化为小值，使初始 ΔW ≈ 0（对齐标准 LoRA 中 B=0 的初始化）
        self.Lambda = nn.Parameter(torch.full((max_rank,), 0.01, **factory))
        self.Q = nn.Parameter(torch.empty(max_rank, in_f, **factory))
        nn.init.kaiming_uniform_(self.P, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.Q, a=5 ** 0.5)
        # 活跃掩码：哪些奇异值参与前向
        self.register_buffer("active_mask", torch.ones(max_rank, dtype=torch.bool, device=base.weight.device))

    @torch.no_grad()
    def set_active_mask(self, mask: torch.Tensor):
        """设置活跃奇异值掩码。"""
        self.active_mask.copy_(mask.to(self.active_mask.device))

    def forward(self, x):
        y = self.base(x)
        # 仅使用活跃的奇异值
        active_lambda = self.Lambda * self.active_mask
        # ΔW = P @ diag(Λ) @ Q, 等价于 (P * Λ) @ Q
        update = F.linear(F.linear(self.dropout(x), self.Q), self.P * active_lambda.unsqueeze(0))
        return y + update * self.scaling

    @torch.no_grad()
    def get_singular_importance(self) -> torch.Tensor:
        """计算每个奇异值的敏感度重要性: |∂L/∂Λ_i| · |Λ_i|"""
        if self.Lambda.grad is None:
            return torch.zeros(self.max_rank, device=self.Lambda.device)
        return (self.Lambda.grad.abs() * self.Lambda.abs()).detach()


class AdaLoRAAllocator:
    """全局奇异值预算分配器。

    在所有层的所有奇异值中，保留重要性最高的 budget 个，其余置零。
    """

    def __init__(self, layers: Dict[str, AdaLoRALinear], budget: int):
        if not layers:
            raise ValueError("no AdaLoRA layers found")
        self.layers = layers
        self.budget = budget
        self.total_singular = sum(l.max_rank for l in layers.values())
        if budget > self.total_singular:
            raise ValueError(f"budget {budget} exceeds total singular values {self.total_singular}")
        self.history: List[dict] = []

    def initialize_uniform(self):
        """均匀激活：每层激活 floor(budget/n_layers) 个奇异值。"""
        n = len(self.layers)
        per_layer = self.budget // n
        remainder = self.budget % n
        for i, layer in enumerate(self.layers.values()):
            mask = torch.zeros(layer.max_rank, dtype=torch.bool)
            active = per_layer + (1 if i < remainder else 0)
            mask[:active] = True
            layer.set_active_mask(mask)
        return {n: int(l.active_mask.sum().item()) for n, l in self.layers.items()}

    @torch.no_grad()
    def reallocate(self, step: int):
        """收集所有奇异值的重要性，全局保留 top-budget。"""
        all_imp = []
        for name, layer in self.layers.items():
            imp = layer.get_singular_importance()
            for i in range(layer.max_rank):
                all_imp.append((name, i, imp[i].item()))

        # 按重要性降序排列
        all_imp.sort(key=lambda x: x[2], reverse=True)

        # 保留 top-budget
        active_set = set()
        for idx, (name, i, _) in enumerate(all_imp):
            if idx < self.budget:
                active_set.add((name, i))

        # 更新各层掩码
        active_counts = {}
        for name, layer in self.layers.items():
            mask = torch.zeros(layer.max_rank, dtype=torch.bool, device=layer.active_mask.device)
            for i in range(layer.max_rank):
                if (name, i) in active_set:
                    mask[i] = True
            layer.set_active_mask(mask)
            active_counts[name] = int(mask.sum().item())

        self.history.append({"step": step, "active_counts": active_counts})
        return active_counts


def inject_adaloora(model: nn.Module, target_names=("q_proj", "v_proj"),
                    max_rank=16, alpha=32.0, dropout=0.05):
    """将匹配的 nn.Linear 替换为 AdaLoRALinear，返回 name->layer 映射。"""
    found = {}

    def visit(parent, prefix=""):
        for name, child in list(parent.named_children()):
            if isinstance(child, AdaLoRALinear):
                continue
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(k in name or k in full for k in target_names):
                wrapped = AdaLoRALinear(child, max_rank, alpha, dropout)
                setattr(parent, name, wrapped)
                found[full] = wrapped
            else:
                visit(child, full)

    visit(model)
    return found
