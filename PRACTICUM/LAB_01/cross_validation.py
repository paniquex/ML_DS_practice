import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds=5):
    """
    params:
        * n - objects amount in sample
        * n_folds - folds amount
    return values:
        * list with size n_folds, where every element is tuple of two 1D numpy array:
            * first array contains indices of train samples
            * second array contains indices of validation samples
    """

    indices = np.arange(n)
    folds = np.array_split(indices, n_folds)
    train_test_idx_list = []
    for i, fold in enumerate(folds):
        test_idx = fold
        train_idx = folds[:i] + folds[i + 1:]
        train_test_idx_list.append((train_idx, test_idx))
    return train_test_idx_list


def accuracy(y_valid, y_true):
    return np.sum(y_valid == y_true, axis=0) / len(y_valid) * 100


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    """
    :param X: train samples
    :param y: targets for train
    :param k_list: list of values of neighbors amount, in ascending order
    :param score: metric name( accuracy' must have)
    :param cv: list of tuples, which contains indices of train and valid samples
    :param kwargs: parameters for __init__ from KNNClassifier
    :return: dict, where keys is neigbors amount from k_list, values - numpy array of size len(cv)
    with accuracy on each fold
    """

    if cv is None:
        cv = kfold(X.shape[0])
    knn = KNNClassifier(**kwargs)
    knn.k = np.max(k_list)
    knn.fit(X, y)
    knn.find_kneighbors(X, return_distance=True)
    accuracy_per_k = {k: np.empty(0) for k in k_list}
    for k in k_list:
        knn.k = k
        for fold in cv:
            y_valid = knn.predict(X[fold[1]])
            accuracy_per_k[k] = np.append(accuracy_per_k[k],
                                          accuracy(y_valid,
                                                   y[fold[1]]))
    return accuracy_per_k