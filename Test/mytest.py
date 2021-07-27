import numpy as np
import time


if __name__ == "__main__":

    # 1
    # a = np.random.randn(100, 100)
    # b = np.random.randn(100, 100)
    #
    # startTime = time.time()
    # for i in range(100):
    #     c2 = np.dot(a, b.T)
    # endTime = time.time()
    # print("dot花费的时间是 : ", (endTime - startTime))
    #
    # startTime = time.time()
    # for i in range(100):
    #     c2 = np.matmul(a, b.T)
    # endTime = time.time()
    # print("matmul花费的时间是 : ", (endTime - startTime))
    #
    # startTime = time.time()
    # for i in range(500):
    #     c1 = np.einsum('ik,jk->ij', a, b, optimize=True)
    # endTime = time.time()
    # print("einsum花费的时间是 : ", (endTime - startTime))
    #
    # print(c1 - c2)

    # 2
    # a = np.random.randn(5)
    # print(a.shape)
    # print(a.T.shape)

    # 3
    # a = np.random.randn(100)
    # print(np.newaxis)
    # st = time.time()
    # for i in range(1000000):
    #     a.reshape(-1, 1)
    # et = time.time()
    # print("reshape花费的时间是 : ", (et-st))
    #
    #
    # st = time.time()
    # for i in range(1000000):
    #     a[:, None]
    # et = time.time()
    # print("newaxis花费的时间是 : ", (et-st))

    # 4
    # a = np.random.randn(5, 5)
    # print(a)
    # print(a[:, None, :])
    # print(a[None, :, :])
    # print(a[:, :, None])

    # 5
    # a = np.random.randn(5)
    # b = np.random.randn(5)
    # print(np.dot(a, b))
    # print(np.matmul(a, b))

    # 6
    # a = np.random.randn(5, 6, 7)
    # b = np.random.randn(6, 10, 15)
    # c = np.einsum('ijk,nml->ijkml', a, b)
    # print(c.shape)

    # 7
    # a = 5
    # b = 4
    # print(a / b)
    # print(type(a))
    # c = 1
    # print(a / c)

    # 8 maxPooling实验
    # data = np.random.randn(256, 128)
    # size = 4
    # data = data.reshape(64, 4, 32, 4)
    # data = data.max(axis=(1, 3))
    # print(data.shape)

    # 9 numpy.repeat
    # eta = np.random.randn(3, 4)
    # print(eta)
    # eta = eta.repeat(2, axis=0).repeat(2, axis=1)
    # print(eta)

    # 10
    # print(2 / 4 ** 2)

    # 11
    # a = np.random.randn(5)
    # b = np.random.randn(5)
    # print(a**2)
    # print(a[0]**2)

    # 12
    # a = np.random.randn(100, 100)
    # st = time.time()
    # for i in range(100000):
    #     np.einsum('ij->', a**2, optimize=True)
    # et = time.time()
    # print(et - st)
    #
    # st = time.time()
    # for i in range(100000):
    #     np.einsum('ij,ij->', a, a, optimize=True)
    # et = time.time()
    # print(et - st)

    # 13
    # a = np.random.randn(10, 20)
    # b = np.zeros((10, 20))
    # b[1, :] = a[9, :]
    # print(b[1])
    # print(a[9])
    # a[9][0] = 1516
    # print(b[1])
    # print(a[9])

    # 14
    # a = (1,2,3,7)
    # b = (1,2,3,7)















    pass