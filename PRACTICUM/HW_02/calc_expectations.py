import numpy as np


def calc_expectations(h, w, X, Q):
    Q_padded = np.pad(Q, pad_width=[[h - 1, 0], [w - 1, 0]],
                      mode='constant',
                      constant_values=-1)
    Q_el_sz = Q.strides[-1]
    Q_pdd_sz0 = Q_padded.shape[0]
    Q_pdd_sz1 = Q_padded.shape[1]
    Q_windows = np.lib.stride_tricks.as_strided(Q_padded,
                                                strides=(Q_el_sz,
                                                         Q_el_sz *
                                                         Q_pdd_sz1,
                                                         Q_el_sz),
                                                shape=(Q_pdd_sz0 *
                                                       Q_pdd_sz1,
                                                       h,
                                                       w))
    not_prob_value = -2
    mask = np.full((h, w), not_prob_value)
    mask[-1][-1] = -1
    cols_to_delete = np.argwhere(Q_windows == mask)[:, 0]
    Q_windows = np.delete(Q_windows, cols_to_delete, axis=0)
    elems_amount = Q.shape[0] * Q.shape[1]
    Q_windows_correct = np.where(Q_windows[:elems_amount] == -1,
                                 0,
                                 Q_windows[:elems_amount])
    probabilities_matrix = np.sum(Q_windows_correct, axis=1)
    probabilities_matrix = np.sum(probabilities_matrix, axis=1)
    return np.multiply(X, probabilities_matrix.reshape(Q.shape))
