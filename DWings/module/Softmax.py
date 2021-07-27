import numpy as np
from DWings.module.module import Module

"""
注意Softmax会内嵌在CELoss中，不用单独做为一层。

把softmax和CELoss合并起来，除了数学意义之外，可以大大简化反向传播的公式，利用O(1)的时间计算梯度。
一般也不会单独把Softmax作为一层，
"""

class Softmax(Module):

    def forward(self, x):
        """
        x.shape = (N, C)
        out = max(0, x)
        """

        # 减去最大值防止溢出!
        v = np.exp(x - x.max(axis=-1, keepdims=True))
        # self.y = v   ps: 这里保存输出是为了求导用的，但是softmax不会单独求导，所以就注释了。
        return v / v.sum(axis=-1, keepdims=True)

    # 一般来说不要用，把softmax和CELoss合起来，反向传播公式太简单太好！！！
    # 一开始写的时候没考虑批量，所以不确定这里的代码能否适应批量数据。
    def backward(self, eta):
        print("warning: softmax backward is used alone")
        k = np.sum(eta * self.y)
        return self.y * (eta - k)
