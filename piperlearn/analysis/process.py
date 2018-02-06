import pickle
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import norm


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

