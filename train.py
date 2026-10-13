import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dynamic_lora import DynamicLoRALinear, DynamicRankAllocator, AllocationConfig

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class TinyLM(nn.Module):
    def __init__(self, vocab=96, d=128, layers=3, max_len=128):
        super().__init__(); self.emb=nn.Embedding(vocab,d); self.pos=nn.Embedding(max_len,d)
        self.blocks=nn.ModuleList([nn.ModuleDict({'linear':nn.Linear(d,d),'ff':nn.Linear(d,d)}) for _ in range(layers)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,vocab); self.max_len=max_len
    def forward(self, x):
        h=self.emb(x)+self.pos(torch.arange(x.size(1),device=x.device)[None,:])
        for b in self.blocks: h=h+torch.tanh(b['linear'](h)); h=h+torch.relu(b['ff'](h))
        return self.head(self.norm(h))

def batch_data(n, seq_len, vocab, device):
    # A learnable modular sequence gives a deterministic language-model task.
    starts=torch.randint(0,vocab,(n,1),device=device); offsets=torch.arange(seq_len+1,device=device)[None,:]
    return (starts+offsets)%vocab

def run(mode, args, out):
    out.mkdir(parents=True, exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed_all(args.seed)
    model=TinyLM().to(device); layers={}
    if args.seq_len + 1 > model.max_len:
        raise SystemExit(f'--seq-len must be <= {model.max_len - 1} (TinyLM max_len={model.max_len}), got {args.seq_len}')
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and name.startswith('blocks'):
            parent=model; parts=name.split('.')
            for p in parts[:-1]: parent=getattr(parent,p)
            old=getattr(parent,parts[-1]); wrapped=DynamicLoRALinear(old,args.max_rank,args.initial_rank,args.alpha,args.dropout)
            setattr(parent,parts[-1],wrapped); layers[name]=wrapped
    # DynamicLoRALinear is created after the base model move; move its LoRA
    # parameters and frozen base weights to the selected device as well.
    model.to(device)
    for p in model.parameters(): p.requires_grad_(False)
    for l in layers.values(): l.lora_A.requires_grad_(True); l.lora_B.requires_grad_(True)
    allocator=DynamicRankAllocator(layers, AllocationConfig(args.rank_budget,args.min_rank,args.max_rank,args.rank_step,args.switch_threshold,args.ema_decay))
    # Start both methods from the same budget-balanced allocation.
    allocator.initialize_uniform()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    vocab=96; losses=[]; start=time.time()
    for step in range(1,args.steps+1):
        x=batch_data(args.batch_size,args.seq_len,vocab,device); logits=model(x[:,:-1]); loss=nn.functional.cross_entropy(logits.reshape(-1,vocab),x[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        if mode=='dynamic': allocator.observe()
        opt.step()
        if mode=='dynamic' and step>=args.warmup_steps and step%args.allocation_interval==0: allocator.rebalance(step, update_importance=False)
        losses.append(float(loss.item()))
    elapsed=time.time()-start; result={'mode':mode,'device':str(device),'final_loss':losses[-1],'mean_last_20':float(np.mean(losses[-20:])),'steps_per_sec':args.steps/elapsed,'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),'rank_history':allocator.history}
    torch.save({'model':model.state_dict(),'ranks':{n:l.rank for n,l in layers.items()},'config':vars(args)},out/'checkpoint.pt')
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'losses.json').write_text(json.dumps(losses),encoding='utf-8'); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['tiny'],default='tiny'); p.add_argument('--mode',choices=['dynamic','uniform','compare'],default='dynamic'); p.add_argument('--steps',type=int,default=300); p.add_argument('--output',default='runs/tiny'); p.add_argument('--seed',type=int,default=42); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--seq-len',type=int,default=64); p.add_argument('--lr',type=float,default=2e-3); p.add_argument('--rank-budget',type=int,default=16); p.add_argument('--min-rank',type=int,default=1); p.add_argument('--max-rank',type=int,default=8); p.add_argument('--rank-step',type=int,default=1); p.add_argument('--initial-rank',type=int,default=2); p.add_argument('--alpha',type=float,default=32); p.add_argument('--dropout',type=float,default=.05); p.add_argument('--allocation-interval',type=int,default=25); p.add_argument('--warmup-steps',type=int,default=25); p.add_argument('--switch-threshold',type=float,default=.5); p.add_argument('--ema-decay',type=float,default=.95); a=p.parse_args()
    if a.steps < 1 or a.batch_size < 1 or a.seq_len < 1: raise SystemExit('steps, batch-size and seq-len must be positive')
    if a.min_rank < 1 or a.max_rank < a.min_rank or a.rank_step < 1: raise SystemExit('rank bounds are invalid')
    if a.allocation_interval < 1 or a.warmup_steps < 0: raise SystemExit('allocation-interval must be positive and warmup-steps non-negative')
    if a.rank_budget < 1: raise SystemExit('rank-budget must be positive')
    if not 0 <= a.ema_decay < 1 or a.switch_threshold < 0: raise SystemExit('EMA decay and switch threshold are invalid')
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    if a.mode=='compare':
        results=[run('uniform',a,root/'uniform'),run('dynamic',a,root/'dynamic')]; (root/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8'); print(json.dumps(results,indent=2))
    else: print(json.dumps(run(a.mode,a,root),indent=2))
if __name__=='__main__': main()
