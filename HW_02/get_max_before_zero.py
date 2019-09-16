import numpy as np


def get_max_before_zero(x):
    zero_positions = np.where(x == 0)[0]
    if (len(zero_positions) == 0) | (zero_positions[0] == (len(x) - 1)):
        return None
    else:
        if (len(x) - 1) == zero_positions[-1]:
            zero_positions = zero_positions[:-1]
        return np.max(x[zero_positions+1])
