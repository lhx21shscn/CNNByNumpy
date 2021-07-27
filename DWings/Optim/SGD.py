import numpy as np

'''
随机梯度下降
'''
class SGD(object):

    def __init__(self, parameters, lr, decay=0.0):
        '''
        :param parameters:网络参数
        :param lr: learning rate
        :param decay: 正则化系数
        '''
        self.parameters = parameters
        self.lr = lr
        self.decay_rate = 1 - decay

    def update(self):
        for item in self.parameters:
            if item.requires_grad:
                if self.decay_rate < 1:
                    item.data *= self.decay_rate
                item.data -= self.lr * item.grad