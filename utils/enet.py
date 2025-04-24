from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def fit_enet(X: np.ndarray, Y: np.ndarray, y_range: np.ndarray, test_index=0,
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


    # set up test, validation and train set
    test_filter = np.repeat(False, nsample)
    test_filter[test_index] = True

    X_test = X[test_index, :].reshape(1, -1)
    y_test = Y[test_index]

    #TODO: remove the following lines that exclude any samples which have the same age as test sample from the training set
    validation_filter = np.repeat(False, nsample)
    validation_filter[Y == y_test] = True # samples that have the same y value as the test sample
    validation_filter[test_index] = False

    y_range_train = y_range[y_range != y_test]
    ## randomly pick one sample stratified by unique values in the training data
    extra_validation_index = [np.random.choice(np.where(Y == value)[0]) for value in y_range_train]
    validation_filter[extra_validation_index] = True

    X_validation = X[validation_filter, :]
    y_validation = Y[validation_filter]

    train_filter = np.isin(Y, y_range)
    train_filter = np.logical_and.reduce((train_filter, np.logical_not(validation_filter), np.logical_not(test_filter)))

    X_train = X[train_filter, :]
    y_train = Y[train_filter]

    train_error = np.zeros(len(lambdas))
    validation_error = np.zeros(len(lambdas))
    y_test_pred = np.zeros(len(lambdas))
    coefficients = np.zeros((len(lambdas), nfeature))
    for id, penalty in enumerate(lambdas):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(alpha=penalty, l1_ratio=0.5, max_iter=20000))
        ])
        model.fit(X_train, y_train)
        coefficients[id, :] = model['elasticnet'].coef_
        y_train_pred = model.predict(X_train)
        train_error[id] = np.mean(np.abs(y_train - y_train_pred))
        y_validation_pred = model.predict(X_validation)
        validation_error[id] = np.mean(np.abs(y_validation - y_validation_pred))
        y_test_pred[id] = model.predict(X_test)

    choice = np.argmin(validation_error)
    best_lambda = lambdas[choice]
    y_test_pred_val = y_test_pred[choice]
    coefficient = coefficients[choice, :]

    return best_lambda, y_test_pred_val, coefficient



def loo_fit(X: np.ndarray, Y: np.ndarray, y_range: np.ndarray, test_indices,
            lambdas=0.2*(2**np.arange(0, 6))):

    nfeature = X.shape[1]
    best_lambda = np.zeros(len(test_indices))
    test_pred = np.zeros(len(test_indices))
    coefs = np.zeros((len(test_indices), nfeature))

    for index, j in enumerate(test_indices):
        # print(j)
        result = fit_enet(X=X, Y=Y, y_range=y_range, test_index=j, lambdas=lambdas)
        best_lambda[index] = result[0]
        test_pred[index] = result[1]
        coefs[index, :] = result[2]

    test_truth = Y[test_indices]
    prediction_result = pd.DataFrame({"Truth": test_truth,
                                      "Predicted": test_pred})

    return prediction_result, coefs, best_lambda

