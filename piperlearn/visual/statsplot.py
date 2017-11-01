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


def plot_single(data, target, figsize, column=None, original=True, percentage=False, sorted=False, abs=False,
                horizontal=True, precision=2, **kwargs):
    col = transform_display(abs, column, data, sorted, target, original)
    total = 0

    if horizontal:
        ax = col.plot.barh(figsize=figsize)
        if percentage:
            totals = []
            for i in ax.patches:
                totals.append(i.get_width())
            total = sum(totals)

        for p in ax.patches:
            if percentage:
                display = str(round((p.get_width()/total) * 100), precision) + '%'
            else:
                display = str(round((p.get_width(), precision)))
            if p.get_width() > 0:
                ax.text(p.get_width() + 0.01, p.get_y() + 0.15, display, **kwargs)
            else:
                ax.text(p.get_width() - 0.01, p.get_y() + 0.15, display, **kwargs)
    else:
        ax = col.plot.bar(figsize=figsize)
        if percentage:
            totals = []
            for i in ax.patches:
                totals.append(i.get_height())
            total = sum(totals)

        for p in ax.patches:
            if percentage:
                display = str(round((p.get_height()/total) * 100), precision) + '%'
            else:
                display = str(round((p.get_height(), precision)))
            if p.get_height() > 0:
                ax.text(p.get_x() - 0.01, p.get_height() + 0.15, display, **kwargs)
            else:
                ax.text(p.get_x() - 0.01, p.get_height() + 0.15, display, **kwargs)


def transform_display(abs, column, data, sorted, target, original):
    dataset = check_dataframe(data, column)
    check_cols(dataset, target)
    if sorted:
        if abs:
            dataset['sorted'] = dataset[target].abs()
        else:
            dataset['sorted'] = dataset[target]
        dataset.sort(columns='sorted', inplace=True)
        dataset.drop('sorted', axis=1)
    if original:
        return dataset[target]
    return dataset[target].value_counts()



