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


def sum_squares(matrix: np.ndarray, vec: np.ndarray, weights=None):
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

    return SSE


def var_comp(geneexp_df, groups):
    """
    :param geneexp_df: sample-by-gene
    :param groups: group labels for all the samples
    :return: R^2 and sorted for all the genes
    """
    geneexp_df_copy = geneexp_df.copy()
    genes = geneexp_df_copy.columns.tolist()
    geneexp_df_copy["Group"] = groups
    geneexp_mean = meanexp(geneexp_df_copy)
    geneexp_mean = geneexp_mean[:-1]
    TSS = sum_squares(geneexp_df_copy.iloc[:, :-1].to_numpy(), geneexp_mean) # total sum of squares

    geneexp_meanbygroup = meanexp(geneexp_df_copy, group="Group")
    count_bygroup = group_counts(geneexp_df_copy, group="Group")

    ESS = sum_squares(geneexp_meanbygroup, geneexp_mean, weights=count_bygroup) # explained sum of squares
    RSS = TSS - ESS # residual sum of squares
    Rsquare = ESS / TSS
    # F_stat = (ESS / (len(count_bygroup)-1)) / (RSS / (len(geneexp_df_copy) - len(count_bygroup)))

    SSdf = pd.DataFrame({"TSS": TSS,
                         "RSS": RSS,
                         "ESS": ESS,
                         "R2": Rsquare})
    SSdf.index = genes
    # SSdf = SSdf.sort_values(by=["Fstat"], ascending=False)

    return SSdf

