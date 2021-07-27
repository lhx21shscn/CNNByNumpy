from DWings.module.Net_QuicklyStart import QSNet
from DWings.loss import CrossEntropyLoss
from DWings.Optim.SGD import SGD
import numpy as np
import time

def train(net, loss_fn, batch_size, optimizer, train_data, train_label):

    dataSize = len(train_data)

    for epoch in range(100):
        print('epoch: ', epoch)

        i = 0
        while i <= dataSize - batch_size:

            data = train_data[i:i + batch_size, :]
            label = train_label[i:i + batch_size, :]
            output = net.forward(data)
            acc, loss = loss_fn(output, label)
            eta = loss_fn.gradient
            net.backward(eta)
            optimizer.update()
            i += batch_size

    print('finsh!')

def eval_acc(net, loss_fn, test_data, test_label):
    # 训练只有10000内存足够可以这么写
    net.nn[2].is_train = False
    net.nn[5].is_train = False
    output = net.forward(test_data)
    acc, loss = loss_fn(output, test_label)
    return acc

if __name__ == '__main__':

    config = [
        {'type': 'Linear', 'in_features': 784, 'out_features': 200, 'bias': True, 'requires_grad': True},
        {'type': 'Relu'},
        {'type': 'Dropout', 'p': 0.5},
        {'type': 'Linear', 'in_features': 200, 'out_features': 100, 'bias': True, 'requires_grad': True},
        {'type': 'Relu'},
        {'type': 'Dropout', 'p': 0.5},
        {'type': 'Linear', 'in_features': 100, 'out_features': 10, 'bias': True, 'requires_grad': True},
    ]

    loss_fn = CrossEntropyLoss()
    net = QSNet(config)
    batch_size = 64
    optimizer = SGD(net.parameter, 0.02)

    train_img = np.loadtxt("./test_data") / 255.
    train_label = np.loadtxt("./test_label")
    print(train_img.shape)
    print(train_label.shape)

    # train
    st = time.time()
    train(net, loss_fn, batch_size, optimizer, train_img, train_label)
    et = time.time()
    print('训练时间是: ',et - st)

    # test
    acc = eval_acc(net, loss_fn, train_img, train_label)
    print(acc)

