import pandas as pd
import os


def check_file(path, file):
    filepath = os.path.join(path, file)
    if not os.path.exists(path):
        os.makedirs(path)
    return filepath


def check_dataframe(data, cols=None):
    if isinstance(data, pd.DataFrame):
        return pd
    return pd.DataFrame(data, columns=cols)

def check_cols(data, column):
    if isinstance(data, pd.DataFrame):
        if column not in list(data.columns.values):
            raise ValueError(column + " is not a valid column in the dataset.")

    raise TypeError("It is not a Pandas DataFrame type.")