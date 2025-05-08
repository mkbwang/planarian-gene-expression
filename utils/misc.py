import re
import numpy as np

# extract the month numbers
def extract_number(mystring):
    numbers = re.findall("^\d+", mystring)
    return int(numbers[0])


def mae(truth, predicted, type="mean"):

    if type not in ["mean", "median"]:
        raise ValueError("Argument must be one of 'mean', 'median'")
    error = 0
    if type == "mean":
        error = np.mean(np.abs(truth - predicted))
    else:
        error = np.median(np.abs(truth-predicted))

    return error
