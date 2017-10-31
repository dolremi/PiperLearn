import seaborn as sns
import matplotlib.pyplot as plt

from ..utility.validation import check_dataframe, check_cols

def compute_correlation(data=None, cols=None, method='pearson'):
    datasets = check_dataframe(data, cols)
    return datasets.corr(method)

class correlations(object):
    def __init__(self, data, cols=None):
        self.corr = compute_correlation(data, cols)

    def plot_heatmap(self, figsize, annot=False, fmt='.2g'):
        plt.figure(figsize=figsize)
        sns.heatmap(self.corr, vmin=-1, vmax=1, annot=annot, fmt=fmt, square=True)

    def plot_individual(self, figsize, target):
        plot_single(self.corr, target, figsize, sorted=True, abs=True, horizontal=True)


def plot_single(self, data, target, figsize, column=None, sorted=False, abs=False, horizontal=True):
    dataset = check_dataframe(data, column)
    check_cols(dataset, target)
    if sorted:
        if abs:
            dataset['sorted'] = dataset[target].abs()
        else:
            dataset['sorted'] = dataset[target]
        dataset.sort(columns='sorted', inplace=True)
        dataset.drop('sorted', axis=1)
    if horizontal:
        ax = dataset[target].plot.barh(figsize=figsize)
        for p in ax.patches:
            if p.get_width() > 0:
                ax.text(p.get_width() + 0.01, p.get_y() + 0.15, str(round((p.get_width()), 4)), color='dimgrey')
            else:
                ax.text(p.get_width() - 0.01, p.get_y() + 0.15, str(round((p.get_width()), 4)), color='dimgrey')
    else:
        ax = dataset[target].plot.bar(figsize=figsize)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.text(p.get_x() - 0.01, p.get_height() + 0.15, str(round((p.get_height()), 4)), color='dimgrey')
            else:
                ax.text(p.get_x() - 0.01, p.get_height() + 0.15, str(round((p.get_height()), 4)), color='dimgrey')



