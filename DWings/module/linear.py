from DWings.module.module import Module
import numpy as np
from DWings.parameter import Parameter

class Linear(Module):
    # in_features : int, out_features : int
    def __init__(self, in_features: int, out_features: int, bias: bool = True, requires_grad: bool = True):

        self.in_features = in_features
        self.out_features = out_features

        W = np.random.randn(in_features, out_features) * (2 / in_features**0.5)
        self.W = Parameter(W, requires_grad)
        if bias:
            b = np.zeros(out_features)
            self.b = Parameter(b, requires_grad)
        self.requires_grad = requires_grad
        self.bias = bias

    def forward(self, x : np.ndarray) -> np.ndarray:
        '''
        :param x: 2D数组，shape:(batch_size, data)
        :return: 2D数组
        '''
        # 保留输入，用以对于参数W的更新。
        if self.requires_grad: self.x = x

        res = np.matmul(x, self.W.data)

        if self.bias:
            res += self.b.data
        return res


    def backward(self, eta : np.ndarray) -> np.ndarray:

        # 批量！
        batch_size = eta.shape[0]

        # 参数更新部分：
        if self.requires_grad:

            self.W.grad = np.matmul(self.x.T, eta) / batch_size
            self.b.grad = np.einsum('ij->j', eta) / batch_size if self.bias else None

        # 梯度回传部分：
        # 优化前： res = np.dot(eta, self.W.T)
        # 如果不涉及转置 einsum比dot要慢，但是涉及转置后，einsum快一点。
        # res = np.einsum('i,ji->j', eta, self.W)
        # 测试发现einsum好像无论如何都不如matmul快！
        res = np.matmul(eta, self.W.data.T)
        return res



