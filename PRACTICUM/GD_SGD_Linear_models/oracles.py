import numpy as np
import scipy
import scipy.special as spec


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef=0, use_bias_in_reg=False):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """

        self.l2_coef = l2_coef
        self.use_bias_in_reg = use_bias_in_reg

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        size = X.shape[0]
        tmp = X.dot(-w.T) * y
        log_value = np.logaddexp(0, tmp).sum()
        if self.use_bias_in_reg:
            reg_value = self.l2_coef * np.linalg.norm(w) ** 2 / 2
        else:
            reg_value = self.l2_coef * np.linalg.norm(w[1:]) ** 2 / 2
        return log_value / size + reg_value

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        size = X.shape[0]
        log_grad_1 = spec.expit(-X.dot(w) * y)
        if self.use_bias_in_reg:
            log_grad = X.T.dot(-y * log_grad_1)
            reg_grad = self.l2_coef * w
            result = log_grad / size + reg_grad
        else:
            log_grad = X.T.dot(-y * log_grad_1)
            reg_grad = self.l2_coef * w
            result = log_grad / size + reg_grad
            result[0] = log_grad[0] / size  # не учитываем reg_grad для смещения
        return result
