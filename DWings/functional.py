"""
sigmoid, tanh, relu, 改进后的sigmoid
均以函数的形式去实现来完成在module中的对应层的正向过程，module中的对应层来完成反向传播
"""
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hardsigmoid(x):
    return 4 * sigmoid(x) - 2

def tanh(x):
    ex = np.exp(x)
    esx = np.exp(-x)
    return (ex - esx) / (ex + esx)