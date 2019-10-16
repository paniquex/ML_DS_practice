import numpy as np

def ensemble_models(X_test, estimators_list, coefs_list=None):
    if coefs_list is None:
        coefs_list = np.ones(len(estimators_list))
    preds_full = np.zeros((X_test.shape[0], len(estimators_list)))
    for i, estimator in enumerate(estimators_list):
        print(estimator)
        print(preds_full.shape)
        preds_full[:, i] = estimator.predict(X_test)
    preds_full = preds_full.astype(int)
    preds = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        preds[i] = np.argmax(np.bincount(preds_full[i, :], coefs_list))
    return preds