import torch
from torch.profiler import profile

x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

with profile() as prof:
    for _ in range(100):
        z = torch.matmul(x, y)

print(prof.key_averages().table(sort_by='cpu_time_total'))
