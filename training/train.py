import torch
import torch.nn as nn

model = nn.Linear(10, 2)

x = torch.randn(32, 10)

output = model(x)
print(output)
