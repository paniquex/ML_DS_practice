import oracles
import numpy as np
import scipy
import scipy.special as spec
import time


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function='binary_logistic',
                 step_alpha=1,
                 step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """

        if loss_function == 'binary_logistic':
            if 'l2_coef' in kwargs.keys():
                self.oracle = oracles.BinaryLogistic(l2_coef=kwargs['l2_coef'])
            else:
                self.oracle = oracles.BinaryLogistic()
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.w = None
        self.X = None
        self.y = None
        self.trace = None
        self.history = None

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        self.X = X
        self.y = y
        if w_0 is None:
            self.w = np.random.uniform(0, 1, X.shape[1])  # need to make it with random
        else:
            self.w = w_0
        self.trace = trace
        if self.trace:
            self.history = {'time': [], 'func': []}

        # Algorithm
        iteration = 0
        start_time = time.time()
        func_val_curr = self.get_objective(self.X, self.y)
        grad_val = self.get_gradient(self.X, self.y)
        self.w = (self.step_beta + 1) * self.w - self.step_alpha * grad_val
        if self.trace:
            self.history['time'].append(time.time() - start_time)
            self.history['func'].append(func_val_curr)
        iteration += 1
        func_val_prev = func_val_curr
        func_val_curr = self.get_objective(self.X, self.y)

        while (iteration < self.max_iter) and \
                (np.abs(func_val_curr - func_val_prev) >= self.tol):
            func_val_prev = func_val_curr
            grad_val = self.get_gradient(self.X, self.y)
            self.w = (self.step_beta + 1) * self.w - self.step_alpha * grad_val
            if self.trace:
                self.history['time'].append(time.time() - start_time)
                self.history['func'].append(func_val_curr)
                start_time = time.time()
            iteration += 1
            func_val_curr = self.get_objective(self.X, self.y)

        if self.trace:
            self.history['time'].append(time.time() - start_time)
            self.history['func'].append(func_val_curr)
            return self.history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """

        predictions = X.dot(self.w)
        predictions = np.where(predictions > 0, 1, -1)
        return predictions

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """

        proba = spec.expit(X.dot(self.w))
        return np.array([proba, 1 - proba]).T

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """

        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """

        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """

        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход


        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """

        if loss_function == 'binary_logistic':
            if 'l2_coef' in kwargs.keys():
                self.oracle = oracles.BinaryLogistic(l2_coef=kwargs['l2_coef'])
            else:
                self.oracle = oracles.BinaryLogistic()
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.seed = random_seed
        self.batch_size = batch_size
        self.w = None
        self.X = None
        self.y = None
        self.trace = None
        self.history = None

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """

        np.random.seed(self.seed)
        self.X = X
        self.y = y
        if w_0 is None:
            self.w = np.random.uniform(0, 1, X.shape[1])  # need to make it with random
        else:
            self.w = w_0
        self.trace = trace
        if self.trace:
            self.history = {'epoch_num': [], 'time': [], 'func': [], 'weights_diff': [0]}

        # Algorithm
        iteration = 0
        relative_epoch_num = 0
        start_time = time.time()
        full_indices = np.random.choice(self.X.shape[0] * self.max_iter)

        indices = full_indices[iteration * self.batch_size: (iteration + 1) * self.batch_size]
        X_batch, y_batch = self.X[indices], self.y[indices]
        func_val_curr = self.get_objective(X_batch, y_batch)
        grad_val = self.get_gradient(X_batch, y_batch)
        self.w = (self.step_beta + 1) * self.w - self.step_alpha * grad_val
        if self.trace:
            self.history['epoch_num'].append(relative_epoch_num)
            self.history['time'].append(time.time() - start_time)
            self.history['func'].append(func_val_curr)
            self.history['weights_diff'].append(0)
        iteration += 1
        func_val_prev = func_val_curr
        relative_epoch_num_prev = 0

        indices = full_indices[iteration * self.batch_size:
                               (iteration + 1) * self.batch_size]
        X_batch, y_batch = self.X[indices], self.y[indices]
        func_val_curr = self.get_objective(X_batch, y_batch)
        while (iteration < self.max_iter) and \
                (np.abs(func_val_curr - func_val_prev) >= self.tol):
            start_time = time.time()
            grad_val = self.get_gradient(X_batch, y_batch)
            self.w = (1 + self.step_beta) * self.w - self.step_alpha * grad_val
            if log_freq < ((relative_epoch_num -
                            relative_epoch_num_prev) / self.X.shape[0]):
                relative_epoch_num_prev = relative_epoch_num
                func_val_prev = func_val_curr
                func_val_curr = self.get_objective(X_batch, y_batch)
                if self.trace:
                    self.history['epoch_num'].append(relative_epoch_num)
                    self.history['time'].append(time.time() - start_time)
                    self.history['func'].append(func_val_curr)
                    prev_weights = self.history['weights_diff'][-1]
                    weights_norm = np.linalg.norm(prev_weights - self.w)
                    self.history['weights_diff'].append(weights_norm)
            iteration += 1
            indices = full_indices[iteration * self.batch_size: (iteration + 1) * self.batch_size]
            X_batch, y_batch = self.X[indices], self.y[indices]
        if self.trace:
            return self.history
