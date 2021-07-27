from DWings.module.module import Module
import numpy as np

class Transform(Module):

    """
    用于单纯形状改变的时候，比如图像分类的 CNN中，卷积层到全连接层时。
    """
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, x):
        return x.reshape(self.out_shape)

    def backward(self, eta):
        return eta.reshape(self.in_shape)