import numpy as np


def get_nonzero_diag_product(X):
    diag_X = np.diag(X)
    non_zero_diag_X = diag_X[diag_X != 0]
    if len(non_zero_diag_X) == 0:
        return None
    else:
        return np.prod(non_zero_diag_X)
