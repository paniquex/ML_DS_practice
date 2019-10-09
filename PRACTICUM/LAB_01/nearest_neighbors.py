from sklearn.neighbors import NearestNeighbors
import numpy as np
from distances import euclidean_distance, cosine_distance


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
            distances, \
            neigh_idxs = self.model.kneighbors(X, n_neighbors=self.k)
        else:
            if self.metric == 'euclidean':
                distances = euclidean_distance(X, self.X_train)
                neigh_idxs = np.argsort(distances,
                                        axis=1)[:, :self.k]
                if return_distance:
                    distances = np.sort(distances,
                                        axis=1)[:, :self.k]

            elif self.metric == 'cosine':
                distances = cosine_distance(X, self.X_train)
                neigh_idxs = np.argsort(distances,
                                        axis=1)[:, :self.k]
                if return_distance:
                    distances = np.sort(distances,
                                        axis=1)[:, :self.k]
        if return_distance:
            return distances, neigh_idxs
        return neigh_idxs

    def predict(self, X):
        """
        params:
            * X - test objects

        return values:
            * numpy array with size X.shape[0] of predictions for test objects from X
        """

        if self.test_block_size > X.shape[0]:
            self.test_block_size = X.shape[0]
        preds = np.zeros(X.shape[0])
        split_size = X.shape[0] // self.test_block_size + \
                     int(X.shape[0] % self.test_block_size != 0)
        classes = np.array(np.unique(self.y_train))
        last_idx = 0
        for i, split in enumerate(np.array_split(X, split_size)):
            distances, neigh_idxs = self.find_kneighbors(split, True)
            for j, idx in enumerate(neigh_idxs):
                counts = np.zeros(len(classes))
                for c in classes:
                    if self.weights:
                        weights = 1 / (distances[j] + self.eps)
                        counts[c] = np.sum((self.y_train[idx] == c) * weights)
                    else:
                        counts[c] = np.sum(self.y_train[idx] == c)
                preds[j + i * split.shape[0]] = np.argmax(counts)
        distances, neigh_idxs = self.find_kneighbors(X[last_idx:], True)
        for j, idx in enumerate(neigh_idxs):
            counts = np.zeros(len(classes))
            for c in classes:
                if self.weights:
                    weights = 1 / (distances[j] + self.eps)
                    counts[c] = np.sum((self.y_train[idx] == c) * weights)
                else:
                    counts[c] = np.sum(self.y_train[idx] == c)
                preds[last_idx + j] = np.argmax(counts)
        return preds
    