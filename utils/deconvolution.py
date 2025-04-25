import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import scipy.stats as ss
from .variation import meanexp, sum_squares, var_comp
from sklearn.preprocessing import StandardScaler

solvers.options['show_progress'] = False




def predict(X: np.ndarray,Y: np.ndarray, labels: np.ndarray, gene_weights:np.ndarray = 1):
    """
    :param X: gene expression matrix with nsamples*ngenes as predictors
    :param Y: gene expression vector with length of ngenes. This is the target sample to predict label
    :param labels: labels of the training samples
    :param weights: weights for each gene when fitting deconvolution
    :param penalty: penalty for difference among estimated weights
    :return: estimated weight vector and weighted sum of labels
    """

    nsamples, ngenes = X.shape
    assert ngenes == len(Y), "Dimensions do not match between training and test samples"
    assert nsamples == len(labels), "Dimensions do not match between the gene expression matrix and labels"

    if len(gene_weights) == 1:
        gene_weights = np.ones(ngenes)
    else:
        assert len(gene_weights) == ngenes, "Length of weights does not equal the number of genes"

    #quadratic terms
    XW = X * gene_weights[np.newaxis, :]
    XWX = np.matmul(XW, X.T)
    XWY = np.matmul(XW, Y)
    XWX = matrix(XWX, (nsamples, nsamples), 'd')
    XWY = matrix(XWY, (nsamples, 1), 'd')

    # TODO: if there is penalty, extra changes need to be made to the quadratic terms based on the distance of labels

    # inequality constraints
    G = matrix(np.identity(nsamples)*-1.0, (nsamples, nsamples), 'd')
    h = matrix(np.zeros(nsamples), (nsamples,1), 'd')

    # equality constraints
    A = matrix(np.ones(nsamples), (1,nsamples), 'd')
    b = matrix(1.0)

    solution = solvers.qp(XWX, -XWY, G, h, A, b)
    label_weights = np.array(solution['x']).flatten()
    label_weights[label_weights < 1e-6] = 0
    label_estim = np.sum(labels * label_weights)
    label_map = labels[np.argmax(label_weights)]

    return label_weights, label_estim, label_map



def loo_predict(expression_df: pd.DataFrame, labels, weighted="None", normalize=False):
    """
    :param expression_df: nsamples x ngenes
    :param labels: labels for all the samples
    :param weighted: None (no weighting), R2 (R square) or Fstat (F statistic)
    :param normalize: whether to standardize each column
    :return: the estimated mean label and the weight matrix
    """

    if weighted not in ["None", "R2", "Fstat"]:
        raise ValueError("Argument must be one of 'None', 'R2', 'Fstat'")

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
        weights = np.ones(ngenes)
        if weighted != "None":
            weights_df = var_comp(train_df, groups=train_labels)
            weights = weights_df[weighted].to_numpy()
            weights = weights / np.max(weights)

        train_mat = train_df.to_numpy()
        validation_vec = validation_df.iloc[0, :].to_numpy()

        # fit deconvolution
        label_probs, mean_label, _ = predict(X=train_mat, Y=validation_vec, labels=train_labels,
                                             gene_weights=weights)

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


