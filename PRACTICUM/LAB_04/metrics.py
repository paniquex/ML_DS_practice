import numpy as np


def accuracy(y_valid, y_true):
    return np.sum(y_valid == y_true, axis=0) / len(y_valid)


def get_precision_array(y_valid, y_true):
    precision_array = np.zeros(len(np.unique(y_true)))
    for i in range(len(precision_array)):
        TP = np.sum((y_true == y_valid) & (y_true == i))
        FP = np.sum((y_valid == i) & (y_true != i))
        precision_array[i] = TP / (TP + FP)
    return precision_array


def get_recall_array(y_valid, y_true):
    recall_array = np.zeros(len(np.unique(y_true)))
    for i in range(len(recall_array)):
        TP = np.sum((y_true == y_valid) & (y_true == i))
        FN = np.sum((y_true == i) & (y_valid != i))
        recall_array[i] = TP / (TP + FN)
    return recall_array


def get_specifity_array(y_valid, y_true):
    specifity_array = np.zeros(len(np.unique(y_true)))
    for i in range(len(specifity_array)):
        TN = np.sum((y_valid != i) & (y_true != i))
        FP = np.sum((y_valid == i) & (y_true != i))
        specifity_array[i] = TN / (TN + FP)
    return specifity_array


def get_f1_score_array(y_valid, y_true):
    recall = get_recall_array(y_valid, y_true)
    precision = get_precision_array(y_valid, y_true)
    f1_score_array = 2 * (recall * precision) / (recall + precision)
    return f1_score_array

