import numpy as np
from sklearn.neighbors import NearestNeighbors


def euclidean_distance(X, Y):
    """
    params:
        * X - np.array with size N x D
        * Y - np.array with size M x D
    return values:
        * np.array with size N x M, where [i, j] - euclidean distance between i-th vector from X and
                                                                              j-th vector from Y
    """

    X_sqr = np.sum(X ** 2, axis=1)[:, None]
    Y_sqr = np.sum(Y ** 2, axis=1)
    return np.sqrt(X_sqr - 2 * np.dot(X, Y.T) + Y_sqr)


def cosine_distance(X, Y):
    """
    params:
        * X - np.array with size N x D
        * Y - np.array with size M x D
    return values:
        * np.array with size N x M, where [i, j] - cosine distance between i-th vector from X and
                                                                              j-th vector from Y
    """

    result = np.dot(X, Y.T)
    result = result / np.sqrt(np.sum(X ** 2, axis=1))[:, None]
    result = result / np.sqrt(np.sum(Y ** 2, axis=1))
    return 1 - result


class KNNClassifier:
    def __init__(self,
                 k=10,
                 strategy='my_own',
                 metric='euclidean',
                 weights=False,
                 test_block_size=10000
                 ):
        """
        params:
            * k - amount of nearest neighbours(nn)
            * strategy - nn searching algorithm. Possible values:
                - 'my_own' - self-realization
                - 'brute' - from sklearn.neighbors.NearestNeighbors(algorithm='brute')
                - 'kd_tree' - from sklearn.neighbors.NearestNeighbors(algorithm='kd_tree')
                - 'ball_tree' - from sklearn.neighbors.NearestNeighbors(algorithm='ball_tree')
            * metric:
                - 'euclidean' - euclidean metric
                - 'cosine' - cosine metric
            * weights - bool variable.
                - True - weighted KNN(with distance)
                - False - simple KNN
            * test_block_size - size of test block
        """

        self.X_train = None
        self.y_train = None
        self.distances = None
        self.neigh_idxs = None
        self.model = None
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.eps = 1e-5
        self.weights = weights
        if self.strategy == 'brute':
            self.model = NearestNeighbors(n_neighbors=self.k, algorithm=self.strategy, metric=self.metric)
        elif self.strategy != 'my_own':
            self.model = NearestNeighbors(n_neighbors=self.k, algorithm=self.strategy)
        else:
            self.model = None
        self.test_block_size = test_block_size

    def fit(self, X, y):
        """
        params:
            * X - train data
            * y - targets for train data
        """

        self.y_train = y.astype(int)
        self.k = np.min([self.y_train.shape[0], self.k])
        if self.strategy != 'my_own':
            self.model.fit(X, self.y_train)
        else:
            self.X_train = X

    def find_kneighbors(self, X, return_distance):
        """
        params:
            * X - objects sample
            * return_distance - bool variable

        return values:
            * If return_distance == True:
                * tuple with two numpy array with size (X.shape[0], k), where:
                  [i, j] elem of first array must be the distance between
                  i-th object and his j-th nearest neighbour
                  [i, j] elem of second array must be the index of j-th nearest neighbour to i-th object
            * If return_distance == False:
                * only second array
        """

        if self.strategy != 'my_own':
            self.distances, \
            self.neigh_idxs = self.model.kneighbors(X, n_neighbors=self.k)
        else:
            if self.metric == 'euclidean':
                self.distances = euclidean_distance(X, self.X_train)
                self.neigh_idxs = np.argsort(self.distances,
                                        axis=1)[:, :self.k]
                if return_distance:
                    self.distances = np.sort(self.distances,
                                        axis=1)[:, :self.k]

            elif self.metric == 'cosine':
                self.distances = cosine_distance(X, self.X_train)
                self.neigh_idxs = np.argsort(self.distances,
                                        axis=1)[:, :self.k]
                if return_distance:
                    self.distances = np.sort(self.distances,
                                        axis=1)[:, :self.k]
        if return_distance:
            return self.distances, self.neigh_idxs
        return self.neigh_idxs

    def predict(self, X):
        """
        params:
            * X - test objects

        return values:
            * numpy array with size X.shape[0] of predictions for test objects from X
        """

        self.test_block_size = np.min(self.test_block_size, X.shape[0])
        preds = np.zeros(X.shape[0])
        split_size = X.shape[0] // self.test_block_size + \
                     int(X.shape[0] % self.test_block_size != 0)
        curr_idx = 0
        classes = np.array(np.unique(self.y_train))
        for i, split in enumerate(np.array_split(X, split_size)):
            self.distances, self.neigh_idxs = self.find_kneighbors(split, True)
            for j, idx in enumerate(self.neigh_idxs):
                counts = np.zeros(len(classes))
                for c in classes:
                    if self.weights:
                        weights = 1 / (self.distances[j] + self.eps)
                        counts[c] = np.sum((self.y_train[idx] == c) * weights)
                    else:
                        counts[c] = np.sum(self.y_train[idx] == c)
                preds[j + curr_idx] = np.argmax(counts)
            curr_idx += split.shape[0]
        return preds

    def predict_for_cv(self, X):
        """
        params:
            * X - test objects

        return values:

        """

        preds = np.zeros(X.shape[0])
        classes = np.array(np.unique(self.y_train))
        for j, idx in enumerate(self.neigh_idxs[:, :self.k]):
            counts = np.zeros(len(classes))
            for c in classes:
                if self.weights:
                    weights = 1 / (self.distances[j, :self.k] + self.eps)
                    counts[c] = np.sum((self.y_train[idx] == c) * weights)
                else:
                    counts[c] = np.sum(self.y_train[idx] == c)
            preds[j] = np.argmax(counts)
        return preds


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
        cv = kfold(X.shape[0], 3)
    knn = None
    if 'k' not in kwargs.keys():
        knn = KNNClassifier(k_list[-1], **kwargs)
    else:
        knn = KNNClassifier(**kwargs)
    accuracy_per_k = {k: np.empty(0) for k in k_list}
    for fold in cv:
        knn.k = k_list[-1]
        knn.fit(X[fold[0]], y[fold[0]])
        knn.find_kneighbors(X[fold[1]], return_distance=True)
        for k in k_list:
            knn.k = k
            y_valid = knn.predict_for_cv(X[fold[1]])
            accuracy_per_k[k] = np.append(accuracy_per_k[k],
                                          accuracy(y_valid,
                                                   y[fold[1]]))
    return accuracy_per_k
