
import pandas as pd
import numpy as np
import re

if __name__ == "__main__":

    expression_df = pd.read_csv("tpm.redSum.combat.ww.txt", sep='\t', index_col=0)
    all_timepoints = expression_df.columns.tolist()

    # filter out the timepoints that belong to wildtypes
    wildtype_pattern = "^\d+MO[a-e]"

    regex = re.compile(wildtype_pattern)
    train_sample_names = [selected_time for selected_time in all_timepoints if regex.match(selected_time)]
    test_sample_names = np.setdiff1d(all_timepoints, train_sample_names)

    expression_df_train = expression_df.loc[:, train_sample_names]
    expression_df_test = expression_df.loc[:, test_sample_names]

    expression_df_train.to_csv("data/train_data.csv")
    expression_df_test.to_csv("data/test_data.csv")

