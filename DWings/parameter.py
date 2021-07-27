import numpy as np

class Parameter(object):

    r"""

    Parameter的作用在于：包装一个numpy数组，使层（or others）的参数有统一的形式。

    examples：
    linear层的参数权重矩阵W和偏置bias可以统一被包装成一个Parameter
    conv2D层的参数卷积核kernel可以被包装成一个Parameter

    """
    def __init__(self, data, requires_grad):

        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

    @property
    def T(self):
        return self.data.T