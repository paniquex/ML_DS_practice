import numpy as np
from distances import cosine_distance, euclidean_distance
from nearest_neighbors import KNNClassifier




def kfold(n, n_folds=3):
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
    size_of_one_fold = int(n / n_folds) + (n < n_folds)
    size_with_folds = size_of_one_fold * n_folds
    out_elements_amount = len(indices) - size_with_folds
    train_test_idx_list = []
    for i in range(n_folds):
        test_idx = indices[i * size_of_one_fold:(i + 1) * size_of_one_fold]
        if out_elements_amount > 0:
            test_idx = np.append(test_idx, n - out_elements_amount)
            out_elements_amount -= 1
        train_idx = np.array(list((set(indices) - set(test_idx))))
        train_test_idx_list.append((train_idx, test_idx))
    return train_test_idx_list


def accuracy(y_valid, y_true):
    return np.sum(y_valid == y_true, axis=0) / len(y_valid)


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
    knn = None
    metric_per_k = {k: np.empty(0) for k in k_list}
    for fold in cv:
        if 'k' not in kwargs.keys():
            knn = KNNClassifier(k_list[-1], **kwargs)
        else:
            knn = KNNClassifier(**kwargs)
        knn.fit(X[fold[0]], y[fold[0]])
        knn.find_kneighbors(X[fold[1]], return_distance=True)
        for k in k_list:
            knn.k = k
            y_valid = knn.predict_for_cv(X[fold[1]])
            if score == "accuracy":
                metric_per_k[k] = np.append(metric_per_k[k],
                                            accuracy(y_valid,
                                                   y[fold[1]]))
    return metric_per_k
