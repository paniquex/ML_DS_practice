import numpy as np
import scipy


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

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """

        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        size = X.shape[0]
        log_value = np.logaddexp(0, np.sum(-w[:, None].T * X, axis=1) * y).sum()
        reg_value = self.l2_coef * np.linalg.norm(w) ** 2 / 2
        return log_value / size + reg_value

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        pass
        # size = X.shape[0]
        # log_grad = np.log(y[:, None]) + np.log(X) + np.log(np.sum(-w[:, None].T * X, axis=1) * y * scipy.special.expit(np.sum(-w[:, None].T * X, axis=1) * y))[:, None]
        # print(log_grad)
        # reg_grad = self.l2_coef * w
        #
        # return log_grad / size + reg_grad
