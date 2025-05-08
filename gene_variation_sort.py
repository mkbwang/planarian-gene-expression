import pandas as pd
import numpy as np
from utils.variation import meanexp, group_counts, sum_squares, var_comp
from utils.misc import extract_number
import os


def sort_variation(exp_df, groups):

    log_exp_df = np.log(exp_df)
    rank_exp_df = exp_df.rank()

    raw_var_comp = var_comp(exp_df, groups)
    gene_order_raw = np.argsort(-raw_var_comp['R2'].to_numpy()).argsort()

    log_var_comp = var_comp(log_exp_df, groups)
    gene_order_log = np.argsort(-log_var_comp['R2'].to_numpy()).argsort()

    rank_var_comp = var_comp(rank_exp_df, groups)
    gene_order_rank = np.argsort(-rank_var_comp['R2'].to_numpy()).argsort()

    order_df = pd.DataFrame({"raw_order": gene_order_raw,
                             "log_order": gene_order_log,
                             "rank_order": gene_order_rank})

    order_df.index = exp_df.columns.tolist()

    return order_df


if __name__ == "__main__":

    # read in the expression dataframe
    gene_expressions = pd.read_csv("data/train_data.csv", index_col=0)
    gene_expressions_mat = gene_expressions.to_numpy()
    genenames = np.array(gene_expressions.index.tolist())
    samples = gene_expressions.columns.tolist()

    # extract ages
    ages = np.array([extract_number(timestring) for timestring in samples])
    unique_ages = np.unique(ages)

    # retain genes that are present in all samples
    prevalence = np.mean(gene_expressions_mat > 0, axis=1)
    subset_gene_id = np.where(prevalence == 1)[0]
    subset_genenames = genenames[subset_gene_id]
    gene_expressions = gene_expressions.loc[subset_genenames, :]
    gene_expressions_mat = gene_expressions_mat[subset_gene_id, :]
    gene_expressions = gene_expressions.T
    gene_expressions_mat = gene_expressions_mat.T

    age_range_list = [(2, 42), (4, 42), (6, 42), (2, 23), (4, 23), (6, 23), (2, 18), (4, 18), (6, 18)]

    output_dict = {}
    for agerange in age_range_list:
        sample_mask = np.logical_and(ages >= agerange[0], ages <= agerange[1])
        gene_expressions_subset = gene_expressions.loc[sample_mask, :]
        ages_subset = ages[sample_mask]

        variation_order_df = sort_variation(gene_expressions_subset, ages_subset)
        variation_order_df = variation_order_df.sort_values('rank_order')
        output_dict[f"{agerange[0]}_{agerange[1]}"] = variation_order_df
        # variation_order_df.to_csv(f"data/variation_order_{agerange[0]}_{agerange[1]}.csv",
        #                           sep='\t')

    with pd.ExcelWriter('data/variation_order.xlsx') as writer:
        for key, value in output_dict.items():
            value.to_excel(writer, sheet_name=key)

