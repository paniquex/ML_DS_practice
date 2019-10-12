import numpy as np
from sklearn.neighbors import NearestNeighbors
from distances import cosine_distance, euclidean_distance


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

        # split_size = X.shape[0] // self.test_block_size + \
            # int(X.shape[0] % self.test_block_size != 0)

        # for i, split in np.array_split(X, split_size):
        if self.strategy != 'my_own':
            self.distances, \
            self.neigh_idxs = self.model.kneighbors(X, n_neighbors=self.k)
        else:
            if self.metric == 'euclidean':
                self.distances = euclidean_distance(X, self.X_train)
                self.neigh_idxs = np.argsort(self.distances,
                                        axis=1)[:, :self.k]
                if return_distance:
                    self.distances = self.distances[np.arange(self.distances.shape[0])[:, None],
                                                    self.neigh_idxs]

            elif self.metric == 'cosine':
                self.distances = cosine_distance(X, self.X_train)
                self.neigh_idxs = np.argsort(self.distances,
                                        axis=1)[:, :self.k]
                if return_distance:
                    self.distances = self.distances[np.arange(self.distances.shape[0])[:, None],
                                                    self.neigh_idxs]
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

        if self.test_block_size > X.shape[0]:
            self.test_block_size = X.shape[0]
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
            # counts = np.zeros(len(classes))
            # for c in classes:
            if self.weights:
                weights = 1 / (self.distances[j, :self.k] + self.eps)
                counts = np.bincount(self.y_train[idx],  weights)
            else:
                counts = np.bincount(self.y_train[idx])
            preds[j] = np.argmax(counts)
        return preds
