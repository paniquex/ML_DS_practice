import numpy as np


def replace_nan_to_means(X):
    X_without_nans = X.copy()
    mean_by_column = np.nanmean(X_without_nans, axis=0)
    mean_by_column = np.where(np.isnan(mean_by_column), 0, mean_by_column)
    X_without_nans = np.where(np.isnan(X_without_nans),
                              mean_by_column,
                              X_without_nans)
    return X_without_nans
