# CNNByNumpy
**包名：DWings == Deep Wings** ~~模仿Tensor+flow，Py+Torch~~

### 在DWings包下实现了：

**Module：**

1. 激活函数：Relu，Sigmoid，HardSigmoid，Tanh
2. 层：全连接层Linear，卷积层Conv2D，池化层：Maxpooling/Meanpooling, Softmax分类层（集成在交叉熵损失函数中），
3. 其他：随机失活Dropout，变形：transform，所有层的父类：Module，集成的网络结构：QSNet（QuicklyStartNet）

**Optim：**

1. SGD
2. Adam

**functional:** 各种激活函数，对应层的正向传播过程会调用这里定义的函数。

**parameter:** 包装numpy.ndarray数组，功能和Pytorch的Parameter类似，为梯度建立存储空间。

**loss:** MSELoss\CrossEntropyLoss损失函数

### 在Test包下实现了：
1. **MLP结构如下：**

   ~~~
    config = [
        {'type': 'Linear', 'in_features': 784, 'out_features': 200, 'bias': True, 'requires_grad': True},
        {'type': 'Relu'},
        {'type': 'Dropout', 'p': 0.5},
        {'type': 'Linear', 'in_features': 200, 'out_features': 100, 'bias': True, 'requires_grad': True},
        {'type': 'Relu'},
        {'type': 'Dropout', 'p': 0.5},
        {'type': 'Linear', 'in_features': 100, 'out_features': 10, 'bias': True, 'requires_grad': True},
    ]
    ~~~


2. **CNN 结构如下：**
    
    ~~~
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
  ~~~

