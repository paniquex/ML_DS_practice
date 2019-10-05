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

    result = np.zeros((X.shape[0], Y.shape[0]))
    for i, x_row in enumerate(X):
        result[i, :] = np.sqrt(np.sum((x_row - Y) ** 2, axis=1))
    return result


def cosine_distance(X, Y):
    """
    params:
        * X - np.array with size N x D
        * Y - np.array with size M x D
    return values:
        * np.array with size N x M, where [i, j] - cosine distance between i-th vector from X and
                                                                              j-th vector from Y
    """

    result = np.zeros((X.shape[0], Y.shape[0]))
    for i, x_row in enumerate(X):
        result[i, :] = np.dot(x_row, Y) / (np.linalg.norm(x_row)) / np.linalg.norm(Y, axis=1)
    return result
