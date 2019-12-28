import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
import time


class RandomForestMSE:
    def __init__(self, n_estimators=202, max_depth=None, feature_subsample_size=None,
                 object_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.object_subsample_size = object_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        if self.object_subsample_size is None:
            self.object_subsample_size = X.shape[0]

        indexes_obj_all = np.arange(X.shape[0])
        self.tree_models = []

        for i in range(self.n_estimators):
            indexes_obj_subset = np.random.choice(indexes_obj_all,
                                                  size=self.object_subsample_size,
                                                  replace=True)
            dec_tree = None
            dec_tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                             max_features=self.feature_subsample_size,
                                             **self.trees_parameters)
            dec_tree.fit(X[indexes_obj_subset],
                         y[indexes_obj_subset])
            self.tree_models.append(dec_tree)
            del dec_tree

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        preds = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            preds += self.tree_models[i].predict(X) / self.n_estimators
        return preds


class GradientBoostingMSE:
    def __init__(self, n_estimators=20, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 object_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use learning_rate * gamma instead of gamma

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.feature_subsample_size = feature_subsample_size
        self.object_subsample_size = object_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        if self.object_subsample_size is None:
            self.object_subsample_size = X.shape[0]

        indexes_obj_all = np.arange(X.shape[0])

        self.obj_indexes = []
        self.alpha_arr = []
        self.models_arr = []

        predict_sum = np.zeros(X.shape[0])

        for i in range(self.n_estimators):
            dec_tree = DecisionTreeRegressor(**self.trees_parameters, max_depth=self.max_depth,
                                             max_features=self.feature_subsample_size)
            indexes_obj_subset = np.random.choice(indexes_obj_all,
                                                  size=self.object_subsample_size,
                                                  replace=True)
            # optimize model

            s_i = 2 * (y[indexes_obj_subset] - predict_sum[indexes_obj_subset])

            dec_tree.fit(X[indexes_obj_subset], s_i)
            pred_opt = dec_tree.predict(X[indexes_obj_subset])
            # optimize model coef

            alpha_i = minimize_scalar(lambda alpha_opt: np.mean(
                (- y[indexes_obj_subset] + predict_sum[indexes_obj_subset] + alpha_opt * pred_opt) ** 2),
                                      bounds=(0, 1000),
                                      method='Bounded').x

            self.obj_indexes.append(indexes_obj_subset)

            self.alpha_arr.append(alpha_i * self.learning_rate)
            self.models_arr.append(dec_tree)

            predict_sum += self.models_arr[-1].predict(X[indexes_obj_subset]) * self.alpha_arr[-1]

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        pred = 0
        preds_all = []
        self.rmse_per_iter = []
        for i in range(len(self.models_arr)):
            preds_all.append(self.models_arr[i].predict(X))
        return np.sum(np.array(self.alpha_arr)[:, None] * np.array(preds_all), axis=0)

    def collect_statistics(self, X, y):

        pred = 0
        preds_all = []
        self.statistics = {'time': [], 'rmse': [], 'max_depth': self.max_depth, 'learning_rate': self.learning_rate,
                           'feature_sub_sample_size': self.feature_subsample_size}
        for i in range(len(self.models_arr)):
            start_time = time.time()
            preds_all.append(self.models_arr[i].predict(X))

            pred = np.sum(np.array(self.alpha_arr)[:i + 1, None] * np.array(preds_all), axis=0)
            self.statistics['rmse'].append(np.sqrt(mean_squared_error(y, pred)))
            self.statistics['time'].append(time.time() - start_time)
        return np.sum(np.array(self.alpha_arr)[:, None] * np.array(preds_all), axis=0)
