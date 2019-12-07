import numpy as np

def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    eye_matrix = np.eye(np.size(w))
    return (np.apply_along_axis(function, axis=1, arr=w[:, None] + eps * eye_matrix) -
            function(w)[:, None]).sum(axis=1) / eps