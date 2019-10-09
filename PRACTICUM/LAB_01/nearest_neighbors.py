from sklearn.neighbors import NearestNeighbors
import numpy as np


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
    result /= np.sqrt(np.sum(X ** 2, axis=1))[:, None]
    result /= np.sqrt(np.sum(Y ** 2, axis=1))
    return 1 - result


class KNNClassifier:
    def __init__(self,
                 k=10,
                 strategy='my_own',
                 metric='euclidean',
                 weights=False,
                 test_block_size=100
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

        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.eps = 1e-5
        self.weights = None
        if weights:
            self.weights = 'distance'
        else:
            self.weights = 'uniform'
        if self.strategy != 'my_own':
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

        self.y_train = y
        if self.strategy != 'my_own':
            self.model.fit(X, y)
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
                                             axis=1,
                                             kind='quicksort')[:, :self.k]
                if return_distance:
                    self.distances = np.sort(self.distances,
                                             axis=1,
                                             kind='quicksort')[:, :self.k]

            elif self.metric == 'cosine':
                self.distances = cosine_distance(X, self.X_train),
                self.neigh_idxs = np.argsort(self.distances,
                                             axis=1,
                                             kind='quicksort')[:, :self.k]
                if return_distance:
                    self.distances = np.sort(self.distances,
                                             axis=1,
                                             kind='quicksort')[:, :self.k]

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

        preds = np.zeros(X.shape[0])
        split_size = X.shape[0] // self.test_block_size + \
                     int(X.shape[0] % self.test_block_size != 0)
        for i, split in enumerate(np.array_split(X, split_size)):
            if i != 0:
                del self.distances
                del self.neigh_idxs
            self.find_kneighbors(split, True)
            for j, idx in enumerate(self.neigh_idxs):
                if self.weights == 'distance':
                    counts = np.bincount(self.y_train[idx],
                                         weights=1 / (self.distances[j, :self.k] + self.eps))
                elif self.weights == 'uniform':
                    counts = np.bincount(self.y_train[idx])
                preds[j + i * self.test_block_size] = np.argmax(counts)
        return preds
