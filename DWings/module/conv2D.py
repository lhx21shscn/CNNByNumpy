import numpy as np
from DWings.module.module import Module
from DWings.parameter import Parameter
import time

class Conv2D(Module):

    def __init__(self, in_channels : int, out_channels : int, kernel_size : int,
                 padding = 'VALID', stride = 1, requires_grad = True, bias = True) -> None:
        '''
        :param in_channels: 输入通道数, 卷积核的通道数, 只有做检查输入是否正确时使用。
        :param out_channels: 输出通道数, 卷积核的数量。
        :param kernel_size: 卷积核大小，这里默认是正方形，边长为奇数。
        :param padding: 模式 : {'VALID', 'SAME'}
        :param stride: 步长
        :param requires_grad: 是否有导数，True or False
        :param bias: 是否有偏置，True or False
        '''
        kernel = np.random.randn(out_channels, kernel_size, kernel_size, in_channels)
        self.kernel = Parameter(kernel, requires_grad)
        if bias:
            b = np.zeros(out_channels)
            self.b = Parameter(b, requires_grad)

        self.requires_grad = requires_grad
        self.stride = stride
        self.bias = bias
        self.padding = padding
        self.kernel_size = kernel_size
        # 下面的信息不保存了，如果需要应该从self.kernel中提取
        # self.in_channels = in_channels
        # self.out_channels = out_channels

    def im2col(self, x):
        '''
        x.shape = N, H, W, C
        目标：根据步长和卷积核大小将x进行分组，利用矩阵乘法加速卷积
        '''
        N, H, W, C = x.shape
        new_H = (H - self.kernel_size) // self.stride + 1
        new_W = (W - self.kernel_size) // self.stride + 1
        # 用 empty 更快
        col = np.empty((N, new_H, new_W, self.kernel_size, self.kernel_size, C))
        '''
        现在的代码用einsum优化了，下面的代码是要转换成2维的矩阵（ipad上的推导），
        利用einsum可以不需要转换成2维。

        这里是从3重循环优化过来的！对照着看一下。
        3重循环：
        col = np.empty((N, new_H, new_W, kernel_size * kernel_size * C))
        for i in range(N):
            for j in range(new_H):
                for k in range(new_W):
                    col[i, j, k, :] = x[i, j*stride:j*stride+kernel_size,
                                      k*stride:k*stride+kernel_size, :].reshape(-1)
        col = col.reshape(-1, kernel_size * kernel_size * C)
        2重循环：
        for j in range(new_H):
            for k in range(new_W):
                col[:, j, k, :] = x[:, j*stride:j*stride+kernel_size,
                                  k*stride:k*stride+kernel_size, :].reshape(N, -1)
        col = col.reshape(-1, kernel_size * kernel_size * C)
        '''

        # 利用einsum函数去优化，不需要为了matmul去进行复杂的转置和形变
        for j in range(self.new_H):
            for k in range(self.new_W):
                col[:, j, k, :, :, :] = x[:, j*self.stride:j*self.stride+self.kernel_size,
                                        k*self.stride:k*self.stride+self.kernel_size, :]
        return col

    def conv(self, x, w):
        '''
        x.shape = N, H, W, C
        w.shape = KN, KH, KW, C
        默认： KH == KW
        res.shape = N, new_H, new_W, KN
        推导见ipad
        '''
        KN, KH, KW, C = w.shape
        N, H, W, C = x.shape
        # 上取整（X / P） == 下取整（X - 1/P） + 1   ps:X,P为正整数
        # 原公式为：上取整((H - KH + 1) // stride)
        self.new_H = (H - KH) // self.stride + 1
        self.new_W = (W - KW) // self.stride + 1

        if self.padding == 'SAME':
            p = KH // 2
            # 填充
            x = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')

        col = self.im2col(x)
        self.col = col
        # W = w.reshape(KN, -1).T
        # res = np.matmul(col, W).reshape(N, new_H, new_W, KN)
        res = np.einsum('ijknml,onml->ijko',col, w)
        if self.bias:
            res += self.b.data
        return res


    def forward(self, x):
        '''
        x.shape = (N, H, W, C)
        N : batch_size
        H, W : Height, Weight
        C: channels
        这里和PyTorch的图像格式：(N, C, H, W) 不太一样,原因如下：
           1. numpy的图像格式就是(H, W, C)(opencv)
           2. 反向传播推公式的时候，需要大量的reshape操作，利用HWC格式的reshape运用比较简洁
           3. 速度快

        if kernel_size = (N1, H1, W1, C1)   ps: C1 == C 是必要的条件
        then result_size = (N, new_H, new_W, N1)
        '''
        return self.conv(x, self.kernel.data)

    def backward(self, eta):

        if self.requires_grad:
            self.kernel.grad = np.einsum('...i,...jkl->ijkl', eta, self.col, optimize=True) / eta.shape[0]
            if self.b is not None:
                self.b.grad = np.einsum('...i->i', eta, optimize=True) / eta.shape[0]


        if self.stride > 1:
            temp = np.zeros((eta.shape[0], self.new_H, self.new_W, eta.shape[3]))
            temp[:, ::self.stride, ::self.stride, :] = eta
            eta = temp

        p = self.kernel_size // 2 if self.padding == 'SAME' else self.kernel_size - 1
        eta = np.pad(eta, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')


        res = np.einsum('ijklmn,nlmo->ijko', self.im2col(eta),
                         self.kernel.data[:, ::-1, ::-1, :], optimize=True)

        return res





















# im2col卷积函数正确性的测试！
if __name__ == '__main__':
    pass
    '''
    测试代码是比对了github：https://github.com/leeroee/NN-by-Numpy/blob/master/package/layers/conv.py
    这里是as_stride方式实现卷积，在cpu上会快与im2col但是用上gpu情况相反。
    
    测试说明：
    把链接里的代码ctrl C+V到主函数上面，然后修改一下导入的库，就可以直接运行了。
    测试代码测试了两方面的内容，是否正确和速度，代码是正确的，einsum优化+optimize=Flase
    速度最快比as_stride慢2倍
    '''
    c = Conv2D(3, 20, 3)
    cc = Conv((20, 3, 3, 3))
    img = np.random.randn(4, 256, 256, 3)
    c.kernel.data = np.random.randn(20, 3, 3, 3)
    cc.W.data = c.kernel.data

    st = time.time()
    for i in range(100):
        res1 = c.forward(img)
    et = time.time()
    print('im2col: ', et - st)
    st = time.time()
    for i in range(100):
        res2 = cc.forward(img)
    et = time.time()
    print('as_strided: ', et - st)
    print(res1.shape)
    print(res2.shape)
    print(((res1 - res2) < 0.00000001).all())