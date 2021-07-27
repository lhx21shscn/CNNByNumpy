import numpy as np
from DWings.module.Softmax import Softmax

r'''
本模块记录Loss Function
大致有：均方误差，交叉熵，
'''

class CrossEntropyLoss(object):

    def __init__(self, requires_acc=True):
        # 内置一个softmax
        self.classifier = Softmax()
        # 如果为Ture, 则不仅返回loss，而且返回批量的正确率
        self.requires_acc = requires_acc

    @property
    def gradient(self):
        # 反向传播的起点，提供第一个梯度。
        # 可以优化！ 在这里直接除batch_size,反向传播都不用除了
        return self.x - self.y

    # __call__方法重载（）运算符
    def __call__(self, x, y):
        '''
        :param x 接受的输出,未经过softmax
        :param y: 真实的标签 one-hot向量
        :return: 损失
        '''
        batch_size = x.shape[0]
        x = self.classifier.forward(x)

        self.x = x
        self.y = y
        if self.requires_acc:
            acc = np.sum(np.argmax(y, axis=1) == np.argmax(x, axis=1)) / batch_size
        loss = -1 * np.einsum('ij,ij->', y, np.log(x)) / batch_size
        if self.requires_acc:
            return acc, loss
        else:
            return loss











class MSELOSS(object):

    def __init__(self):
        pass

    def __call__(self, y, label):
        '''
        y.shape = label.shape = (N, K)
        N: batch_size
        K: 输出类别数
        '''
        self.g = y - label
        loss = 0.5 * np.einsum('ij->', self.g**2)
        return loss

    @property
    def gradient(self):
        return self.g
