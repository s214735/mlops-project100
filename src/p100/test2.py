import torch
a = torch.rand(10000, 10000).to('cuda')
b = torch.mm(a, a)
print(b)

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())