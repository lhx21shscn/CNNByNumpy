from DWings.module.module import Module
import numpy as np
import DWings.functional as F

class HardSigmoid(Module):

    def __init__(self):
        pass

    def forward(self, x):
        self.x = F.sigmoid(x)
        return 4 * self.x - 2

    def backward(self, eta):
        # return 4 * eta * self.y * (1 - self.y)
        return 4 * np.einsum('...,...,...->...', self.x, 1 - self.x, eta, optimize=True)