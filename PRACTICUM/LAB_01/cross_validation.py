import numpy as np


def kfold(n, n_folds):
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
    size_of_one_fold = int(len(indices) / n_folds)
    size_with_folds = size_of_one_fold * n_folds
    out_elemnts_amount = len(indices) - size_with_folds
    train_test_idx_list = []
    for i in range(n_folds):
        test_idx = indices[i * size_of_one_fold:(i + 1) * size_of_one_fold]
        if out_elemnts_amount > 0:
            test_idx = np.append(test_idx, n - out_elemnts_amount)
            out_elemnts_amount -= 1
        train_idx = np.array(list((set(indices) - set(test_idx))))
        train_test_idx_list.append((train_idx, test_idx))
    return train_test_idx_list
