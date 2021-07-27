from DWings.module.linear import Linear
from DWings.module.Relu import Relu
from DWings.module.module import Module
from DWings.module.conv2D import Conv2D
from DWings.module.maxpooling import MaxPool
from DWings.module.meanpooling import MeanPool
from DWings.module.transform import Transform
from DWings.module.Dropout import Dropout
from DWings.module.Tanh import Tanh
from DWings.module.Sigmoid import Sigmoid
from DWings.module.HardSigmoid import HardSigmoid


'''
此类仅用于简化网络的实现（QuicklyStartNet）
'''
class QSNet(Module):

    def __init__(self, config):

        self.parameter = []
        self.nn = []
        for layer in config:
            self.nn.append(self.__createLayer__(layer))
    def __createLayer__(self, layer):

        t = layer['type']
        if t == 'Linear':
            in_features = layer['in_features']
            out_features = layer['out_features']
            bias = layer['bias']
            requires_grad = layer['requires_grad']

            res = Linear(in_features, out_features, bias=bias, requires_grad=requires_grad)

            self.parameter.append(res.W)
            if res.b is not None:
                self.parameter.append(res.b)
        elif t == 'Relu':
            res = Relu()
        elif t == 'Sigmoid':
            res = Sigmoid()
        elif t == 'HardSigmoid':
            res = HardSigmoid()
        elif t == 'Tanh':
            res = Tanh()
        elif t == 'Dropout':
            p = layer['p']
            res = Dropout(p)
        elif t == 'Conv2D':
            in_channels = layer['in_channels']
            out_channels = layer['out_channels']
            kernel_size = layer['kernel_size']
            padding = layer['padding']
            stride = layer['stride']
            requires_grad = layer['requires_grad']
            bias = layer['bias']
            # print(in_channels, out_channels, kernel_size, padding, stride, requires_grad, bias)
            res = Conv2D(in_channels, out_channels, kernel_size, padding, stride, requires_grad, bias)
            # res = Conv((out_channels, kernel_size, kernel_size, in_channels), padding, stride, requires_grad, bias)
            self.parameter.append(res.kernel)
            if res.b is not None:
                self.parameter.append(res.b)
        elif t == 'MaxPool':
            size = layer['size']
            res = MaxPool(size)
        elif t == 'MeanPool':
            size = layer['size']
            res = MeanPool(size)
        elif t == 'Transform':
            in_shape = layer['in_shape']
            out_shape = layer['out_shape']
            res = Transform(in_shape, out_shape)
        else:
            raise TypeError("用于建立神经网络的类型参数有误")

        return res


    def forward(self, x):

        for layer in self.nn:
            # print(x.shape)
            x = layer.forward(x)

        return x


    def backward(self, eta):

        # 倒序
        for layer in self.nn[::-1]:
            eta = layer.backward(eta)
        return eta