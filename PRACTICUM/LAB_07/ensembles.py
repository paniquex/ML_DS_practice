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
        if feature_subsample_size is None:
            self.feature_subsample_size = 1
        else:
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
        indexes_feat_all = np.arange(X.shape[1])
        self.tree_models = []
        for i in range(self.n_estimators):
            indexes_obj_subset = np.random.choice(indexes_obj_all,
                                                  size=int(alpha * X.shape[0]),
                                                  replace=False)
            indexes_feat_subset = np.random.choice(indexes_feat_all,
                                                   size=int(self.feature_subsample_size * X.shape[1]),
                                                   replace=False)
            dec_tree = None
            dec_tree = DecisionTreeRegressor(*self.trees_parameters,
                                             max_depth=self.max_depth,
                                             )
            dec_tree.fit(X[indexes_obj_subset[:, None], indexes_feat_subset],
                         y[indexes_obj_subset])
            self.tree_models.append((dec_tree, indexes_feat_subset))
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
        indexes_all = np.arange(X.shape[0])
        for i in range(self.n_estimators):
            preds += self.tree_models[i][0].predict(X[indexes_all[:, None],
                                                      self.tree_models[i][1]]) / self.n_estimators
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
        pass

    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        pass

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pass
