import os

import pandas as pd
from sklearn.model_selection import train_test_split


def _check_file(filename):
    if not os.path.exists(filename):
        raise ValueError("File or Path " + filename + "does not exist.")


def _check_data(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The input data needs to be Pandas DataFrame")


def _check_col(column, data):
    _check_data(data)
    if column not in list(data.columns.values):
        raise ValueError(column + " is not a valid column in the dataset. ")


def prepare_X_y(data, target):
    y = data[target]
    del data[target]
    X = data
    X, y = check_X_y(X, y)
    return X, y


def load_data(filename):
    _check_file(filename)
    return pd.read_csv(filename)


def save_data(data, target, outputfile, random_state=0, split_ratio=0.3, verbose=False):
    _check_col(target, data)
    train, test = train_test_split(data, test_size=split_ratio, random_state=random_state, stratify=data[target])
    if verbose:
        print("The training data has " + str(train.shape[0]) + " samples.")
        print("The test data has " + str(test.shape[0]) + " samples.")
        print("Here is the distribution for the column of " + target + " in training set: ")
        print(train[target].value_counts())
        print("The data types for training set: ")
        print(train.get_dtype_counts())
        print("Here is the distribution for the column of " + target + " in test set: ")
        print(test[target].value_counts())
        print("The data types for test set: ")
        print(test.get_dtype_counts())
    if outputfile[-3:] != 'pkl':
        raise ValueError("The output file should be a pkl file.")
    train_X, train_y = prepare_X_y(train, target)
    test_X, test_y = prepare_X_y(test, target)
    datset = {"train_X": train_X, "train_Y": train_y, "test_X": test_X, "test_Y": test_y}
    with open(outputfile, "wb") as f:
        pickle.dump(datset, f, pickle.HIGHEST_PROTOCOL)


def check_missing(data):
    _check_data(data)
    names = data.columns[data.isnull().any()]
    if len(names):
        print("The column names that have missing values are: " + ",".join(list(names)))
    else:
        print("There are no missing values in this dataset")


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