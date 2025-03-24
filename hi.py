import torch
import numpy as np
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = np.array([[1]])
z = np.array([[1, 2, 3], [4, 5, 6]])

for i, val in enumerate(x[0]):
    print(i, val.item())
