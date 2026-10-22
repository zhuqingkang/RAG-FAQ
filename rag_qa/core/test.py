import torch
import numpy as np

# 假设模型输出
logits = torch.tensor([[2.3, 0.1],
                       [5.5, 3.2]])

# PyTorch 写法
pre_torch = torch.argmax(logits, dim=1)   # tensor([0, 1])  1: 表示行方向 ，0： 表示列方向

# NumPy 写法（把 logits 转成 numpy）
pre_numpy = np.argmax(logits.numpy(), axis=-1)  # array([0, 1])

print(pre_torch)   # tensor([0, 1])
print(pre_numpy)   # [0 1]