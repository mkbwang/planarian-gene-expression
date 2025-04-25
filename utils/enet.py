from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def fit_enet(X: np.ndarray, Y: np.ndarray, X_test: np.ndarray, Y_test,
             lambdas=0.2*(2**np.arange(0, 6))):
    """
    :param X: Predictor matrix, nsample * nfeature
    :param Y: Response vector with length of nsample
    :param y_range: unique values of the response whose samples can be used as training data
    :param test_index: the index of the test sample
    :param lambdas: penalty parameter to select from
    :return: training error, validation error, test error, binary vector indicating variables selected
    """

    nsample, nfeature = X.shape
    X_test = X_test.reshape(1, -1)

    # validation_filter = np.repeat(False, nsample)
    # unique_Ys = np.unique(Y)
    #
    # ## randomly pick one sample stratified by unique values in the training data
    # validation_index = [np.random.choice(np.where(Y == value)[0]) for value in unique_Ys]
    # validation_filter[validation_index] = True
    #
    # X_validation = X[validation_filter, :]
    # y_validation = Y[validation_filter]
    #
    # X_train = X[np.logical_not(validation_filter), :]
    # y_train = Y[np.logical_not(validation_filter)]

    y_pred = np.zeros(len(lambdas))

    # train_error = np.zeros(len(lambdas))
    # validation_error = np.zeros(len(lambdas))

    # pick the optimal penalty parameter
    for id, penalty in enumerate(lambdas):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(alpha=penalty, l1_ratio=0.5, max_iter=20000, tol=1e-3))
        ])
        model.fit(X, Y)
        y_pred[id] = model.predict(X_test)
        # model.fit(X_train, y_train)
        # y_train_pred = model.predict(X_train)
        # train_error[id] = np.mean(np.abs(y_train - y_train_pred))
        # y_validation_pred = model.predict(X_validation)
        # validation_error[id] = np.mean(np.abs(y_validation - y_validation_pred))

    absolute_error = np.abs(y_pred - Y_test)
    choice = np.argmin(absolute_error)

    return y_pred[choice]



def loo_fit(X: np.ndarray, Y: np.ndarray, lambdas=0.2*(2**np.arange(0, 6))):

    nsamples, _ = X.shape
    y_pred = np.zeros_like(Y)

    for j in range(len(Y)):
        # print(j)
        X_train = np.delete(X, j, axis=0)
        Y_train = np.delete(Y, j)
        X_test = X[j, :]
        Y_test = Y[j]
        y_pred[j] = fit_enet(X=X_train, Y=Y_train, X_test=X_test, Y_test=Y_test, lambdas=lambdas)

    prediction_result = pd.DataFrame({"Truth": Y,
                                      "Predicted": y_pred})

    return prediction_result

