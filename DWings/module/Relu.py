from DWings.module.module import Module
import numpy as np
import DWings.functional as F

class Relu(Module):

    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return F.relu(x)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta