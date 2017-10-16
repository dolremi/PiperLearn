import pandas as pd
import os
import pickle
import numpy as np
import scipy.stats as st
from scipy.special import boxcox1p
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
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

def _check_X_y(X, y):
    if not isinstance(X, pd.DataFrame) or X.ndim != 2:
        raise TypeError("X needs to be 2D Pandas DataFrame")
    if not isinstance(y, pd.Series) or y.ndim != 1:
        raise TypeError("y needs to be 1D Pandas Series")
    if X.shape[0] != y.shape[0]:
        raise TypeError("X and y don't have same number of samples.")

def prepare_X_y(data, target):
    y = data[target]
    del data[target]
    X = data
    _check_X_y(X, y)
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

class FeatureBuilder(object):
    def __init__(self, inputfile, output):
        _check_file(inputfile)
        self.inputfile = inputfile
        _check_file(output)
        self.output = output


    def read_data(self, verbose=False):
        data = pickle.load(open(self.inputfile, 'rb'))
        self.train_X = data['train_X']
        self.train_Y = data['train_Y']
        self.test_X = data['test_X']
        self.test_Y = data['test_Y']
        if verbose:
            print("For training dataset: ")
            print(self.train_X.shape)
            print("The distribution of the training set: ")
            print(self.train_Y.value_counts())
            print("For test dataset: ")
            print(self.test_X.shape)
            print("The distribution of the test set: ")
            print(self.test_Y.value_counts())


    def build(self, handler, new_col):
        self.train_X[new_col] = self.train_X.apply(lambda row: handler(row), axis=1)
        self.test_X[new_col] = self.test_X.apply(lambda row: handler(row), axis=1)

    def save_data(self, filename, verbose=False):
        file = self.output + filename
        data = {'train_X': self.train_X, 'train_Y':self.train_Y, 'test_X':self.test_X, 'test_Y': self.test_Y}
        with open(file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        if verbose:
            print("For training dataset: ")
            print(self.train_X.shape)
            print("The distribution of the training set: ")
            print(self.train_Y.value_counts())
            print("For test dataset: ")
            print(self.test_X.shape)
            print("The distribution of the test set: ")
            print(self.test_Y.value_counts())

class UniPlot(object):
    def __init__(self, data, column,size):
        self.data = data
        self.column = column
        self.size = size

    def set_col(self, col):
        _check_col(col, self.data)
        self.column = col

    def set_size(self, size):
        self.size = size

    def plot_counts(self):
        _check_col(self.column, self.data)
        f, ax = plt.subplots(figsize=self.size)
        sns.countplot(y=self.column, data=self.data, color='c')

    def plot_dist(self):
        _check_col(self.column, self.data)
        plt.figure(figsize = self.size)
        sns.distplot(self.data[self.column].dropna(), color = 'r', fit = norm)

    def plot_pp(self):
        _check_col(self.column, self.data)
        fig = plt.figure(figsize = self.size)
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.95, 0.95, "Skewness: %f\n Kurtosis: %f" %(self.data[self.column].skew(),
                                                            self.data[self.column].kurt()), transform=ax.transAxes,
                                                            va="top")
        st.probplot(self.data[self.column], dist="norm", plot=ax)

    def plot_kde(self, target):
        _check_col(target, self.data)
        _check_col(self.column, self.data)
        facet = sns.FacetGrid(self.data, hue=target, aspect=4)
        facet.map(sns.kdeplot, self.column, shade=True)
        facet.set(xlim=(0, self.data[self.column].max()))
        facet.add_legend()

    def boxcox_trans(self, alpha=0.15):
        self.data[self.column] = boxcoxTransform(self.data[self.column], alpha)

    def log_trans(self):
        self.data[self.column] = np.log1p(self.data[self.column])

class Correlations(object):
    def __init__(self, data, target=None):
        if target:
            _check_col(target, data)
            self.target = target
        self.corrolations = data.corr()


    def plot_heatmap(self, size, annot=False, fmt='.2g'):
        plt.figure(figsize=size)
        sns.heatmap(self.corrolations, vmin=-1, vmax=1, annot=annot, fmt=fmt, square=True)

    def plot_corr(self, size, target=None):
        self.corrolations['sorted'] = self.corrolations[self.target].abs()
        self.corrolations.sort(columns='sorted').drop('sorted', axis=1)
        if self.target:
            match_corr = self.corrolations[self.target]
        elif target:
            match_corr = self.corrolations[target]
        else:
            raise ValueError("There is no value for target argument. The name of target column is needed.")
        match_corr.plot.barh(figsize=size)


def boxcoxTransform(column, alpha=0.15):
    column = boxcox1p(column, alpha)
    column += 1
    return column

