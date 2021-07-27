from abc import ABCMeta, abstractmethod

class Module(metaclass=ABCMeta):
    '''
    作为所有层的基类，如果自定义新的层应该从此类继承并重写下面两个方法
    '''

    # 前向传播时，传入参数x，尽量返回重新申请的空间，避免改变了数据。
    @abstractmethod
    def forward(self, *args):
        pass

    # 反向传播时，传入参数eta，尽量返回eta，尽量不申请多余空间。
    @abstractmethod
    def backward(self, *args):
        pass