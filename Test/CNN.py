from DWings.module.Net_QuicklyStart import QSNet
from DWings.loss import CrossEntropyLoss
from DWings.Optim.Adam import Adam
from DWings.Optim.SGD import SGD
import time
import numpy as np


def train(net, loss_fn, batch_size, optimizer, train_data, train_label):

    dataSize = len(train_data)
    print(dataSize)
    for epoch in range(5):
        print('epoch: ', epoch)
        cnt, i = 0, 0
        while i <= dataSize - batch_size:
            data = train_data[i:i + batch_size, :]
            label = train_label[i:i + batch_size, :]
            output = net.forward(data)
            acc, loss = loss_fn(output, label)
            print(loss, acc, cnt)
            eta = loss_fn.gradient
            net.backward(eta)
            optimizer.update()
            i += batch_size
            cnt += 1
    print('finsh !')

def eval_acc(net, loss_fn, test_data, test_label):
    # 训练只有10000内存足够可以这么写
    output = net.forward(test_data)
    acc, loss = loss_fn(output, test_label)
    return acc

if __name__ == '__main__':

    batch_size = 128
    config = [
        {'type':'Conv2D', 'in_channels':1, 'out_channels':8, 'kernel_size':5, 'padding':'VALID', 'stride':1,
         'requires_grad':True, 'bias':True},
        {'type':'Relu'},
        {'type':'MaxPool', 'size':2},
        {'type':'Conv2D', 'in_channels':8, 'out_channels':16, 'kernel_size':5, 'padding':'VALID', 'stride':1,
         'requires_grad':True, 'bias':True},
        {'type':'Relu'},
        {'type':'MaxPool', 'size':2},
        {'type':'Transform', 'in_shape':(-1, 4, 4, 16), 'out_shape':(-1, 256)},
        {'type':'Linear', 'in_features':256, 'out_features':64, 'bias':True, 'requires_grad':True},
        {'type':'Relu'},
        {'type':'Linear', 'in_features':64, 'out_features':10, 'bias':True, 'requires_grad':True}
    ]

    loss_fn = CrossEntropyLoss()
    net = QSNet(config)
    optimizer = Adam(net.parameter, 0.005)

    # 归一化！！！！
    train_data = np.loadtxt("./train_data").reshape(-1, 28, 28, 1) / 255.
    train_label = np.loadtxt("./train_label")
    test_data = np.loadtxt("./test_data").reshape(-1, 28, 28, 1) / 255.
    test_label = np.loadtxt("./test_label")

    # train
    st = time.time()
    train(net, loss_fn, batch_size, optimizer, train_data, train_label)
    et = time.time()
    print('训练时间是: ', et - st)

    # test
    print(eval_acc(net, loss_fn, test_data, test_label))

