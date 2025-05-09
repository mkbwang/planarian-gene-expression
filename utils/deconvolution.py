import pandas as pd
import numpy as np
# from cvxopt import matrix, solvers
import scipy.stats as ss
from .variation import meanexp, sum_squares, var_comp
from sklearn.preprocessing import StandardScaler
import osqp
import scipy.sparse as sparse

# solvers.options['show_progress'] = False



def predict_l2(X:np.ndarray, Y:np.ndarray, labels: np.ndarray, penalty=0, verbose=False):
    """
    :param X: gene expression matrix with nsamples*ngenes as predictors
    :param Y: gene expression vector with length of ngenes. This is the target sample to predict label
    :param labels: labels of the training samples
    :param penalty: penalty parameter for limiting weight differences between samples
    :return: estimated weight vector and weighted sum of labels
    """

    nsamples, ngenes = X.shape
    assert ngenes == len(Y), "Dimensions do not match between training and test samples"
    assert nsamples == len(labels), "Dimensions do not match between the gene expression matrix and labels"

    XXt = np.matmul(X, X.T)
    XY = np.matmul(X, Y).flatten()

    # calculate penalty matrix for samples with similar ages
    labels_mat_1 = np.tile(labels, (nsamples, 1))
    labels_mat_2 = np.tile(labels.reshape(-1, 1), (1, nsamples))
    inverse_abs_distance = penalty*ngenes/(np.abs(labels_mat_1 - labels_mat_2)+1)
    Penaltymat = -inverse_abs_distance
    np.fill_diagonal(Penaltymat,
                     np.sum(inverse_abs_distance, axis=1) - np.diag(inverse_abs_distance))

    Pmat = sparse.csc_matrix(XXt + Penaltymat)
    qvec = -XY

    # linear constraint
    A_0 = np.identity(nsamples)
    l_0 = np.zeros(nsamples)
    u_0 = np.ones(nsamples) * np.inf

    A_1 = np.ones((1, nsamples))
    l_1 = 1
    u_1 = 1

    A = sparse.csc_matrix(np.concatenate((A_0, A_1), axis=0))
    l_vec = np.append(l_0, l_1)
    u_vec = np.append(u_0, u_1)

    model = osqp.OSQP()
    model.setup(Pmat, qvec, A, l_vec, u_vec, verbose=verbose)
    result = model.solve()

    solution = result.x
    solution[solution < 1.0/nsamples/10] = 0
    solution = solution / np.sum(solution)

    label_estim = np.sum(labels * solution)

    return solution, label_estim



def predict_l1(X:np.ndarray, Y:np.ndarray, labels: np.ndarray, penalty=0, verbose=False):
    """
    :param X: gene expression matrix with nsamples*ngenes as predictors
    :param Y: gene expression vector with length of ngenes. This is the target sample to predict label
    :param labels: labels of the training samples
    :param penalty: penalty parameter for limiting weight differences between samples
    :return: estimated weight vector and weighted sum of labels
    """
    nsamples, ngenes = X.shape
    assert ngenes == len(Y), "Dimensions do not match between training and test samples"
    assert nsamples == len(labels), "Dimensions do not match between the gene expression matrix and labels"

    # calculate penalty matrix for samples with similar ages
    labels_mat_1 = np.tile(labels, (nsamples, 1))
    labels_mat_2 = np.tile(labels.reshape(-1, 1), (1, nsamples))
    inverse_abs_distance = penalty*ngenes / (np.abs(labels_mat_1 - labels_mat_2) + 1)
    Penaltymat = -inverse_abs_distance
    np.fill_diagonal(Penaltymat,
                     np.sum(inverse_abs_distance, axis=1) - np.diag(inverse_abs_distance))

    Pdiag = np.zeros(nsamples+ngenes)
    Pmat = np.diag(Pdiag)
    Pmat[0:nsamples, 0:nsamples] += Penaltymat
    Pmat = sparse.csc_matrix(Pmat)

    qvec = np.concatenate((np.zeros(nsamples), np.ones(ngenes)))

    # linear constraints
    A_0 = np.concatenate((np.ones(nsamples), np.zeros(ngenes))).reshape(1, -1)
    l_0 = np.array([1])
    u_0 = np.array([1])
    A_1 = np.concatenate((np.identity(nsamples), np.zeros((nsamples, ngenes))), axis=1)
    l_1 = np.zeros(nsamples)
    u_1 = np.ones(nsamples) * np.inf

    ## constraint for absolute value
    A_2 = np.concatenate((X.T, np.identity(ngenes)), axis=1)
    l_2 = Y
    u_2 = np.ones(ngenes) * np.inf

    A_3 = np.concatenate((-X.T, np.identity(ngenes)), axis=1)
    l_3 = -Y
    u_3 = np.ones(ngenes) * np.inf

    A = sparse.csc_matrix(np.concatenate((A_0, A_1, A_2, A_3), axis=0))
    l_vec = np.concatenate((l_0, l_1, l_2, l_3))
    u_vec = np.concatenate((u_0, u_1, u_2, u_3))

    model = osqp.OSQP()
    model.setup(Pmat, qvec, A, l_vec, u_vec, verbose=verbose)
    result = model.solve()

    solution = result.x[:nsamples]
    solution[solution < 1.0 / nsamples / 10] = 0
    solution = solution / np.sum(solution)

    label_estim = np.sum(labels * solution)

    return solution, label_estim





def loo_predict(expression_df: pd.DataFrame, labels, type="l2", penalty=0,  normalize=True):
    """
    :param expression_df: nsamples x ngenes
    :param labels: labels for all the samples
    :param normalize: whether to standardize each column
    :return: the estimated mean label and the weight matrix
    """

    fit_func = predict_l2 if type == "l2" else predict_l1


    output_df = pd.DataFrame({
        "Truth": labels,
        "Predicted": np.zeros_like(labels)
    })

    nsamples, ngenes = expression_df.shape
    prob_mat = np.zeros((nsamples, nsamples))

    for j in range(nsamples):

        train_ids = np.arange(nsamples)
        train_ids = train_ids[train_ids != j]
        train_df = expression_df.iloc[train_ids, :]
        train_labels = labels[train_ids]
        validation_df = expression_df.iloc[[j]]
        if normalize:
            scaler = StandardScaler()
            train_df = pd.DataFrame(scaler.fit_transform(train_df),
                                    index=train_df.index, columns=train_df.columns)
            validation_df = pd.DataFrame(scaler.transform(validation_df),
                                         index=validation_df.index, columns=validation_df.columns)


        train_mat = train_df.to_numpy()
        validation_vec = validation_df.iloc[0, :].to_numpy()

        # fit deconvolution
        label_probs, mean_label = fit_func(X=train_mat, Y=validation_vec, labels=train_labels,
                                             penalty=penalty)

        label_probs = np.insert(label_probs, j, 0)
        prob_mat[j, :] = label_probs
        output_df["Predicted"][j] = mean_label

    return output_df, prob_mat











# def validation_performance(expression_df: pd.DataFrame, group="Age", predictor="indv", weighted="None", num_predictor=-1):
#
#     """
#     :param expression_df: gene expression matrix with samples on rows and genes on columns. One extra column is the group variable
#     :param group: column name of the group
#     :param predictor: use the individual ("indv") sample for prediction or mean expressions at group level ("group") for prediction
#     :param weighted: whether to assign different weights for each gene when fitting deconvolution
#     :param num_predictor: number of genes to be used for deconvolution, by default -1 (all the features)
#     :return: dataframe with true age and predicted age
#     """
#
#     if predictor not in ['indv', 'group']:
#         raise ValueError("Argument must be one of 'indv' or 'group'")
#
#     if weighted not in ['None', "invvar", "Rsquare"]:
#         raise ValueError("Argument must be one of 'None', 'invvar', 'Rsquare'")
#
#     nsamples = len(expression_df)
#
#     labels = expression_df[group].to_numpy()
#     # expression_df = expression_df.drop(group, axis=1)
#
#     output_df = pd.DataFrame({
#         "Truth": labels,
#         "Prediction": np.zeros_like(labels),
#         "MAP": np.zeros_like(labels)
#     })
#
#     for j in range(nsamples):
#
#         rows_to_keep = list(range(len(expression_df)))
#         rows_to_keep.remove(j)
#         train_df = expression_df.iloc[rows_to_keep, :]
#         train_labels = np.delete(labels, j)
#         unique_labels, counts = np.unique(train_labels, return_counts=True)
#         train_indvexp = train_df.drop(group, axis=1).to_numpy()
#         train_meanexp = meanexp(train_df.drop(group, axis=1), group=None) # overall mean for each gene
#         train_meanexp_group = meanexp(train_df, group=group) # gene mean by group
#         SST = sum_squares(train_indvexp, train_meanexp, weights=None) # total sum of squares
#         SSG = sum_squares(train_meanexp_group, train_meanexp, weights=counts) # sum of squares among groups
#
#
#         Rsquare = SSG / SST
#         variance = SST / (len(train_df) - 1)
#
#         weights = np.ones_like(variance)
#         if weighted == "Rsquare":
#             weights = Rsquare / variance
#         elif weighted == "invvar":
#             weights = 1 / variance
#
#         # target
#         validation_df = expression_df.iloc[j, :].drop(group)
#         validation_vec = validation_df.to_numpy().flatten()
#
#         if num_predictor != -1:
#             ranks = ss.rankdata(-Rsquare)
#             subset_genes = np.where(ranks <= num_predictor)[0]
#             train_indvexp = train_indvexp[:, subset_genes]
#             train_meanexp_group = train_meanexp_group[:, subset_genes]
#             validation_vec = validation_vec[subset_genes]
#             weights = weights[subset_genes]
#
#
#
#         if predictor == 'indv':
#             label_weights, label_estim, label_map = predict(X=train_indvexp, Y=validation_vec,
#                                                  labels=train_labels, gene_weights=weights)
#         else:
#             label_weights, label_estim, label_map = predict(X=train_meanexp_group, Y=validation_vec,
#                                                  labels=unique_labels, gene_weights=weights)
#
#         output_df.iloc[j, 1] = label_estim
#         output_df.iloc[j, 2] = label_map
#
#     return output_df


