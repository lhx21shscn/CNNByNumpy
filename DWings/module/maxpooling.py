import numpy as np
from DWings.module.module import Module

'''
MaxPool的推导虽然非常简单，但是如何高效的实现是很难的，这里灵活运用了reshape、max、repeat
去掉了for循环，实现的比较好。

需要理解4维数组(如下)的存储结构

—————                —————
|  ———        ———        |   
| |   | .... |   |...    |
|  ———        ———        |
|   .          .         |
|   .          .         |
|   .          .         |
—————                —————
'''

class MaxPool(Module):

    def __init__(self, size):
        # 池化的窗口大小
        self.size = size

    def forward(self, x):
        '''
        x.shape = (N, H, W, C)
        利用 reshape、max、repeat，去除了for循环，实现的非常简洁，需要对4（高）维数组有很深的理解。
        y.shape = (N, H//size, W//size, C)
        '''
        N, H, W, C = x.shape
        y = x.reshape(N, H//self.size, self.size, W//self.size, self.size, C)
        y = y.max(axis=(2, 4))
        # 保存不是最大值的位置
        self.index = (y.repeat(self.size, axis=1).repeat(self.size, axis=2) != x)
        return y

    def backward(self, eta):
        '''
        eta.shape = (N, H, W, C)
        return.shape = (N, H*size, W*size, C)
        '''
        eta = eta.repeat(self.size, axis=1).repeat(self.size, axis=2)
       
        eta[self.index] = 0
        return eta