import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
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
        self.trees_parameters = trees_parameters

    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        alpha = 1

        indexes_obj_all = np.arange(X.shape[0])
        self.tree_models = []
        for i in range(self.n_estimators):
            indexes_obj_subset = np.random.choice(indexes_obj_all,
                                                  size=int(alpha * X.shape[0]),
                                                  replace=False)
            dec_tree = None
            dec_tree = DecisionTreeRegressor(*self.trees_parameters,
                                             max_depth=self.max_depth,
                                             max_features=self.feature_subsample_size
                                             )
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
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
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
        self.trees_parameters = trees_parameters

    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        dec_tree = DecisionTreeRegressor(**self.trees_parameters, max_depth=self.max_depth,
                                         max_features=self.feature_subsample_size)
        dec_tree.fit(X, y)
        indexes_obj_all = np.arange(X.shape[0])
        alpha = 1

        self.models_arr = [dec_tree]
        self.alpha_arr = [1]
        self.obj_indexes = []

        for i in range(self.n_estimators - 1):
            indexes_obj_subset = np.random.choice(indexes_obj_all,
                                                  size=int(alpha * X.shape[0]),
                                                  replace=False)
            self.obj_indexes.append(indexes_obj_subset)
            # optimize model
            s_i = 0
            pred = 0
            for i in range(len(self.models_arr)):
                pred += self.models_arr[i].predict(X[self.obj_indexes[i]]) * self.alpha_arr[i]
                s_i += 2 * (y[self.obj_indexes[i]] - self.models_arr[i].predict(X[self.obj_indexes[i]]))

            dec_tree.fit(X[indexes_obj_subset], s_i)

            # optimize model coef
            alpha_i = minimize_scalar(lambda alpha_opt: (y[indexes_obj_subset] - pred - alpha_opt * dec_tree.predict(
                X[indexes_obj_subset])) ** 2,
                                      bounds=[0, 1e3])
            self.alpha_arr.append(alpha_i)
            self.models_arr.append(dec_tree)

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
        for i in range(len(self.models_arr)):
            pred += self.models_arr.predict(X[self.obj_indexes[i]]) * self.alpha_arr[i]
        return pred