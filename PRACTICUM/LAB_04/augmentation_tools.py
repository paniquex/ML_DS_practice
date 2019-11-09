import cross_validation
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata

# image preprocessing:
import scipy.ndimage as ndimage
import skimage.transform as transform


def made_transformation(img,
                        type_of_transformation='rotation',
                        param_of_transformation=5):
    if type_of_transformation == 'rotation':
        img_augmented = transform.rotate(img, param_of_transformation, resize=False)
    elif type_of_transformation == 'shift':
        img_augmented = ndimage.shift(img, param_of_transformation)
    elif type_of_transformation == 'blur':
        img_augmented = ndimage.filters.gaussian_filter(img, param_of_transformation)
    elif type_of_transformation == 'all':
        img_augmented = transform.rotate(img, param_of_transformation[0], resize=False)
        img_augmented = ndimage.shift(img_augmented, param_of_transformation[1])
        img_augmented = ndimage.filters.gaussian_filter(img_augmented, param_of_transformation[2])
    elif type_of_transformation == 'equal':
        img_augmented = img
    return img_augmented


def made_augmentation(X,
                      y,
                      new_objects_amount=10000,
                      type_of_transformation='rotation',
                      param_of_transformation=5,
                      ):
    new_objects_amount = np.min([X.shape[0], new_objects_amount])
    if type_of_transformation == 'rotation':
        X_augmented = np.apply_along_axis(
                                            lambda x: transform.rotate(x.reshape(28, 28),
                                                                       param_of_transformation,
                                                                       resize=False),
                                            axis=1,
                                            arr=X[:new_objects_amount])
    elif type_of_transformation == 'shift':
        X_augmented = np.apply_along_axis(
                                            lambda x: ndimage.shift(x.reshape(28, 28),
                                                                    param_of_transformation),
                                            axis=1,
                                            arr=X[:new_objects_amount])
    elif type_of_transformation == 'blur':
        X_augmented = np.apply_along_axis(
                                            lambda x: ndimage.filters.gaussian_filter(x.reshape(28, 28),
                                                                                      param_of_transformation ** (1/2)),
                                            axis=1,
                                            arr=X[:new_objects_amount])
    elif type_of_transformation == 'all':
        X_augmented = np.apply_along_axis(
                                            lambda x: transform.rotate(x.reshape(28, 28),
                                                                       param_of_transformation[0],
                                                                       resize=False),
                                            axis=1,
                                            arr=X[:new_objects_amount])
        X_augmented = X_augmented.reshape(-1, 784)
        X_augmented = np.apply_along_axis(
                                            lambda x: ndimage.shift(x.reshape(28, 28),
                                                                    param_of_transformation[1]),
                                            axis=1,
                                            arr=X_augmented[:new_objects_amount])
        X_augmented = X_augmented.reshape(-1, 784)
        X_augmented = np.apply_along_axis(
            lambda x: ndimage.filters.gaussian_filter(x.reshape(28, 28),
                                                      param_of_transformation[2] ** (1 / 2)),
            axis=1,
            arr=X_augmented[:new_objects_amount])
    elif type_of_transformation == 'all_parallel':
        X_augmented1 = np.apply_along_axis(
                                            lambda x: transform.rotate(x.reshape(28, 28),
                                                                       param_of_transformation[0],
                                                                       resize=False),
                                            axis=1,
                                            arr=X[:new_objects_amount])
        X_augmented2 = np.apply_along_axis(
                                            lambda x: ndimage.shift(x.reshape(28, 28),
                                                                    param_of_transformation[1]),
                                            axis=1,
                                            arr=X[:new_objects_amount])
        X_augmented3 = np.apply_along_axis(
                                            lambda x: ndimage.filters.gaussian_filter(x.reshape(28, 28),
                                                                                      param_of_transformation[2] ** (1/2)),
                                            axis=1,
                                            arr=X[:new_objects_amount])
        X_augmented = np.concatenate([X_augmented1, X_augmented2, X_augmented3], axis=0)
    elif type_of_transformation == 'equal':
        X_augmented = np.array([])
    X_augmented = X_augmented.reshape(-1, 784)
    X_augmented = np.concatenate([X, X_augmented], axis=0)
    y_augmented = y
    if y is not None:
        if type_of_transformation == 'all_parallel':
            y_augmented = np.concatenate([y, y[:new_objects_amount], y[:new_objects_amount], y[:new_objects_amount]])
        else:
            y_augmented = np.append(y, y[:new_objects_amount])

    return X_augmented, y_augmented


def find_best_value_of_transformation(X_train,
                                      y_train,
                                      new_objects_amount=10000,
                                      type_of_transformation='rotation',
                                      param_list_of_transformation=(0, 5, 10, 15),
                                      k_neighbors=4,
                                      metric='cosine',
                                      strategy='brute',
                                      weights=True,
                                      k_folds=3
                                      ):
    best_score = -1
    best_param = -1
    if (type_of_transformation != 'all') & (type_of_transformation != 'all_parallel'):
        for param_of_transformation in param_list_of_transformation:
            preds_per_fold = \
                cross_validation.knn_cross_val_score_with_aug_for_train(X=X_train, y=y_train,
                                                                        new_objects_amount=new_objects_amount,
                                                                        type_of_transformation=type_of_transformation,
                                                                        param_of_transformation=param_of_transformation,
                                                                        score='accuracy',
                                                                        k_list=[k_neighbors],
                                                                        metric=metric,
                                                                        strategy=strategy,
                                                                        weights=weights,
                                                                        k_folds=k_folds
                                                                        )
            print('Score on ' + str(param_of_transformation) + ' param value',
                  np.mean(preds_per_fold[k_neighbors]))
            if np.mean(preds_per_fold[k_neighbors]) > best_score:
                best_score = np.mean(preds_per_fold[k_neighbors])
                best_param = param_of_transformation
    else:
        preds_per_fold = \
            cross_validation.knn_cross_val_score_with_aug_for_train(X=X_train, y=y_train,
                                                                    new_objects_amount=new_objects_amount,
                                                                    type_of_transformation=type_of_transformation,
                                                                    param_of_transformation=param_list_of_transformation,
                                                                    score='accuracy',
                                                                    k_list=[k_neighbors],
                                                                    metric=metric,
                                                                    strategy=strategy,
                                                                    weights=weights,
                                                                    k_folds=k_folds
                                                                    )
        print('Score on ' + str(param_list_of_transformation) + ' param value',
              np.mean(preds_per_fold[k_neighbors]))

    return best_score, best_param


def test_time_augmentation(estimator,
                           X_test,
                           type_of_transformation_list,
                           param_of_transformation_list
                           ):
    """
    tta - for test_time_augmentations amount with different augs on test
    """

    X_test_augmented, _ = made_augmentation(X_test,
                                            y=None,
                                            type_of_transformation=type_of_transformation_list,
                                            param_of_transformation=param_of_transformation_list
                                            )
    print(X_test_augmented.shape)
    distances_tta = np.zeros((X_test.shape[0],
                              estimator.k * X_test_augmented.shape[0] // X_test.shape[0]))
    idxs_tta = np.zeros((X_test.shape[0],
                              estimator.k * X_test_augmented.shape[0] // X_test.shape[0]))
    for i, split in enumerate(np.array_split(X_test_augmented,
                                             X_test_augmented.shape[0] // X_test.shape[0])):
        distances_tta[:, i * estimator.k:(i+1) * estimator.k], \
            idxs_tta[:, i * estimator.k:(i+1) * estimator.k] = estimator.find_kneighbors(split, True)
    neigh_idxs_relative = np.argsort(distances_tta,
                                     axis=1)[:, :estimator.k]
    estimator.neigh_idxs = idxs_tta[np.arange(idxs_tta.shape[0])[:, None],
                                   neigh_idxs_relative.astype(int)].astype(int)
    estimator.distances = distances_tta[np.arange(distances_tta.shape[0])[:, None],
                                         neigh_idxs_relative.astype(int)]
    preds = estimator.predict_for_cv(X_test)
    return preds
