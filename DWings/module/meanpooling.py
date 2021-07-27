import numpy as np
from DWings.module.module import Module

# 非常少用
class MeanPool(Module):

    def __init__(self, size):
        # 窗口大小
        self.size = size

    def forward(self, x):
        N, H, W, C = x.shape
        y = x.reshape(N, H//self.size, self.size, W//self.size, self.size)
        y = y.mean(axis=(2, 4))
        return y

    def backward(self, eta):
        eta = (1 / self.size ** 2) * eta.repeat(self.size, axis=1).repeat(self.size, axis=2)
        return eta