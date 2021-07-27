from DWings.module.module import Module
import numpy as np
import DWings.functional as F

class Sigmoid(Module):

    def __init__(self):
        pass

    def forward(self, x):
        self.x = F.sigmoid(x)
        return self.x

    def backward(self, eta):
        # return eta * self.y * (1 - self.y)
        return np.einsum('...,...,...->...', self.x, 1 - self.x, eta, optimize=True)