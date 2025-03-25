import pandas as pd
import numpy as np


def meanexp(expression_df: pd.DataFrame, group=None) -> np.ndarray:
    """
    :param expression_df: gene expression dataframe with nsamples*(ngenes+1), extra column corresponds to group variable
    :param group: the group variable
    :return: mean gen eexpresson for each gene
    """
    if group is None:
        mean_expression = expression_df.mean().to_numpy()
    else:
        mean_expression = expression_df.groupby(group).mean().to_numpy()

    return mean_expression

def group_counts(expression_df, group="Age") -> np.ndarray:
    """
    :param expression_df: gene expression dataframe with nsamples*(ngenes+1), extra column corresponds to group variable
    :param group: group variable
    :return: counts of sample for each unique group type
    """
    count_vec = expression_df.groupby(group).size().to_numpy()
    return count_vec


def sum_squares(matrix: np.ndarray, vec: np.ndarray, weights=None, sd=True):
    """
    :param matrix: expression matrix with nsamples*ngenes
    :param vec: vector of length ngenes
    :param weights: weight for each sample, length of nsample
    :param sd: whether to calculate the standard deviation
    :return: sum of squares or standard deviation for each gene
    """
    difference = matrix - vec[np.newaxis,:]
    sq_difference = np.square(difference)

    if weights is not None:
        SSE = np.sum(sq_difference * weights[:, np.newaxis], axis=0)
    else:
        SSE = np.sum(sq_difference, axis=0)

    if sd:
        standard_deviation = np.sqrt(SSE / (matrix.shape[0]-1))
        return standard_deviation
    else:
        return SSE

