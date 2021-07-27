import numpy as np
from DWings.module.module import Module

class Dropout(Module):
    def __init__(self, p, is_train=True):
        """
        :param p: 丢弃概率
        公式：   x = 0              概率为 p
         or     x = x / (1 - p)    otherwise
        """
        self.dropout = p
        self.is_train = is_train

    def forward(self, x):

        if self.is_train:
            if self.dropout == 1.0:
                return np.zeros(x.shape)
            if self.dropout == 0.0:
                return x

            mat = (np.random.rand(*x.shape) > self.dropout)
            self.mat = mat
            return mat * x / (1 - self.dropout)

        else:
            return x


    def backward(self, eta):

        if self.is_train:
            if self.dropout == 1.0:
                eta[:] = 0
                return eta
            if self.dropout == 0.0:
                return eta

            return self.mat * eta / (self.dropout)

        else:
            raise ValueError("在反向传播时，仍然正在进行梯度下降")


