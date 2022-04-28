import torch
import numpy

a = torch.zeros([16,25])
b = a.detach().numpy().max(axis=1)
print(b)

a = torch.ones(5)
b = a.detach().numpy()
print(b)
