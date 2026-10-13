"""检查 CUDA PyTorch 是否能在当前电脑完成一次矩阵计算。"""
import torch

if not torch.cuda.is_available():
    raise SystemExit('CUDA 不可用，请安装 CUDA 版 PyTorch')
device=torch.device('cuda')
a=torch.randn(2048,2048,device=device)
b=torch.randn(2048,2048,device=device)
c=a@b
torch.cuda.synchronize()
print('CUDA OK')
print('torch:',torch.__version__)
print('cuda:',torch.version.cuda)
print('device:',torch.cuda.get_device_name(0))
print('result_mean:',float(c.mean()))
