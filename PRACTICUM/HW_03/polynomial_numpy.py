import numpy as np


class Polynomial:
    def __init__(self, *args):
        if len(args) == 0:
            self.coefs = 0
            self.idxs = 0
        else:
            self.coefs = np.array(args)
            self.idxs = np.arange(self.coefs.shape[0])
        
    def __call__(self, x):
        return np.sum(x ** self.idxs * self.coefs)
