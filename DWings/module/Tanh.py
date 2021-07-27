from DWings.module.module import Module
import numpy as np
import DWings.functional as F

class Tanh(Module):

    def __init__(self):
        pass

    def forward(self, x):
        self.x = F.tanh(x)
        return self.x

    def backward(self, eta):
        # return eta * (1 + self.y) * (1 - self.y)
        return np.einsum('...,...,...->...', 1 - self.x, 1 + self.x, eta, optimize=True)