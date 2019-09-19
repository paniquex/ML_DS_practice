import numpy as np


def encode_rle(x):
    if len(x) > 0:
        idx_with_differences = np.append(np.where(np.diff(x) != 0)[0],
                                         np.argwhere(x == x[-1])[-1])
        return (x[idx_with_differences], np.diff(np.insert(idx_with_differences, 0, -1)))
    else:
        return None
