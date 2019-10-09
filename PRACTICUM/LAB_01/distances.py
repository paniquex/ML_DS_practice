import numpy as np


def euclidean_distance(X, Y):
    """
    params:
        * X - np.array with size N x D
        * Y - np.array with size M x D
    return values:
        * np.array with size N x M, where [i, j] - euclidean distance between i-th vector from X and
                                                                              j-th vector from Y
    """

    X_sqr = np.sum(X ** 2, axis=1)[:, None]
    Y_sqr = np.sum(Y ** 2, axis=1)
    return np.sqrt(X_sqr - 2 * np.dot(X, Y.T) + Y_sqr)


def cosine_distance(X, Y):
    """
    params:
        * X - np.array with size N x D
        * Y - np.array with size M x D
    return values:
        * np.array with size N x M, where [i, j] - cosine distance between i-th vector from X and
                                                                              j-th vector from Y
    """

    result = np.dot(X, Y.T)
    result /= np.sqrt(np.sum(X ** 2, axis=1))[:, None]
    result /= np.sqrt(np.sum(Y ** 2, axis=1))
    return 1 - result