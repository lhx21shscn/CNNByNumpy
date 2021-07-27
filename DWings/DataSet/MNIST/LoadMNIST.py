import os
import struct
import numpy as np

'''
把ubyte形式的MNIST手写数字数据集放在同一文件夹下，直接运行可以得到训练和测试数据。
data.shape : (784,)
label.shape : (10,) one-hot向量
'''

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    if kind == 'test':
        kind = 't10k'
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def transformToTxt(path):
    Set, Label = load_mnist('', path)
    print(Set.shape, Label.shape)
    label = np.zeros([len(Label), 10])
    for i in range(len(Label)):
        label[i][Label[i]] = 1
    np.savetxt(path + '_label', label)
    np.savetxt(path + '_data', Set)
    print('finsh! ' + path)

if __name__ == '__main__':
    transformToTxt('train')
    transformToTxt('test')