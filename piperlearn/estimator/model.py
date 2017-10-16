import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, validation_curve, learning_curve, \
    RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    auc, make_scorer


def check_file(filePath):
    """
    Check if the file or folder exists.
    :param filePath: file or folder
    :return: None
    """
    if not os.path.exists(filePath):
        raise ValueError("File " + filePath + " does not exists.")

def check_estimator(estimator):
    """
    Check if the estimaotr has fit and predict method
    :param estimator: The predictive model
    :return: None
    """
    fit_method = getattr(estimator, "fit", None)
    if not callable(fit_method):
        raise TypeError(estimator.__name__ + " does not has a fit method.")
    predict_method = getattr(estimator, "predict", None)
    if not callable(predict_method):
        raise TypeError(estimator.__name__ + " does not has a predict method.")


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_roc_curve(y_test, y_pred):
    """
    Plot ROC curve given the prediction variables and response variables
    :param y_test: The response variables
    :param y_pred: The prediction variables
    :return: None
    """
    fpr, tpr, threholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f) ' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc='lower right')
    plt.show()

def run_model(estimator, train_X=None, train_Y=None, test_X=None):
    """
    Train the data and predict the result using the estimator.
    :param estimator: The predictive model
    :param train_X: The predictor variables for training
    :param train_Y: The response variables for training
    :param test_X: The predictor variables for testing
    :return: The predicted variables
    """
    check_estimator(estimator)
    if train_X is not None and train_Y is not None:
        estimator.fit(train_X, train_Y)
    return estimator.predict(test_X)


class Model(object):
    """
    The class Model can read in the training and test dataset then do the parameter tuning and output the corresponding
    results.
    """
    def __init__(self, inputfile, outputfolder):
        """
        Initialize the input file and output folder and check if they exits.
        :param inputfile: A pkl file contains the training and test data.
        :param outputfolder: A folder that will output the results.
        """
        check_file(inputfile)
        self.input = inputfile
        check_file(outputfolder)
        self.output = outputfolder

    def set_model(self, estimator):
        """
        Set the individual predictive model after validating it.
        :param estimator: A predictive model which has 'fit' and 'predict' method.
        :return: None
        """
        check_estimator(estimator)
        self.model = estimator

    def load_data(self, verbose=False):
        """
        Load the data from a pkl file and pkl file needs to has the format like this:
             data['train_X'] has the predictor variables for training
             data['train_Y'] has the response variables for training
             data['test_X'] has the predictor variables for test
             data['test_Y'] has the response variables for test
        :param verbose: if some summary statistic need to be output.
        :return: None
        """
        if not self.input[-3:] == 'pkl':
            raise TypeError("The input file needs to be a pkl file.")
        data = pickle.load(open(self.input, 'rb'))
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

    def search_param(self, score_metrics, param_grid, splits=5, jobs=-1, verbose=150,n_iters=None,
                     estimator=LogisticRegression(random_state=1, n_jobs=-1, class_weight='balanced')):
        """
        Search the parameters for Logistic Regression Model and StandardScaler piepeline.
        For parameter_grid, all keys should start with `classify` and if n_iters is specified, a randomizedSearch method
        willl be used.
        :param score_metrics: The performance metrics is used for search.
        :param param_grid: A dictionary with key starts with 'classify' for search grid.
        :param splits: The number of splits for cross_validation
        :param jobs: Number of parallel jobs.
        :param verbose: The verbose level to print out the steps.
        :param n_iters: Number of sampled candidates or sampling iterations for computing budget for randomized search.
        :param estimator: The predictive model
        :return: None
        """
        for param in param_grid:
            for key in param:
                if not key.startswith('classify'):
                    print('For keys in param_grid, they need to start with classify.')
                    return

        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1)
        pipe = Pipeline([('standardized', StandardScaler()),
                         ('classify', estimator)])
        if n_iters:
            gs = RandomizedSearchCV(pipe, param_grid, n_iter=n_iters, scoring=make_scorer(score_metrics),
                                    n_jobs=jobs, verbose=verbose, random_state=1)
        else:
            gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=make_scorer(score_metrics),
                          n_jobs=jobs, verbose=verbose)
        gs.fit(self.train_X, self.train_Y)
        self.model = gs.best_estimator_
        self.train_individual(self.test_X, self.test_Y)



    def train_individual(self, X, y, roc=False, fit=True):
        """
        Run the estimator for the data
        :param X: The predictor variables
        :param y: The response variables
        :param roc: boolean variable to indicate if ROC curve needs to plot
        :param fit: If the model needs to be fit
        :return: None
        """
        if fit:
            pred_Y = run_model(self.model, self.train_X, self.train_Y, X)
        else:
            pred_Y = run_model(self.model, test_X=X)
        self.precision = precision_score(y, pred_Y)
        self.recall = recall_score(y, pred_Y)
        self.f1 = f1_score(y, pred_Y)
        self.report = classification_report(y, pred_Y)
        self.threhold = self.model.predict_proba(X)[:, 1]
        self.prob = 50
        if roc:
            plot_roc_curve(y, pred_Y)

    def show_result(self):
        """
        Print out the precision, recall and f1 score and current threshold for the probability.
        """
        print("Now the best estimator is ")
        print(self.model)
        print("The precision is " + str(self.precision))
        print("The recall is " + str(self.recall))
        print("F1 score is " + str(self.f1))
        print("Now the threshold for probability is " + str(self.prob) + "%.")
        print("Now the classification report is ")
        print(self.report)

    def show_coef(self, filename=None, verbose=False):
        """
        Save the parameter coefficient of the model
        :param filename: The output file name.
        :param verbose: Whether to output the coefficient.
        """
        num_features = self.test_X.shape[1]
        coef = []
        if hasattr(self.model, 'coef_'):
            features = self.model.coef_
        else:
            features = self.model.named_steps['classify'].coef_
        for i in range(num_features):
            coef.append(features[0][i])
        coefficient = pd.Series(coef, index=self.test_X.columns)
        coefficient.sort_values(inplace=True)
        if verbose:
            print(coefficient)
        if filename:
            new_file = os.path.join(self.output, filename)
            coefficient.to_csv(new_file)
        else:
            print("No data has been saved.")

    def modify_threshold(self, threshold, Y, filename=None, X=None, printing=False, save=False):
        """
        Modify the threshold value and save the false positive and false negative samples.
        :param threshold: The threshold value
        :param Y: The response variables
        :param filename: The general filename
        :param X: The predictor variables
        :param printing: Whether to output the FP, FN, TP, TN info.
        """
        self.prob = threshold
        total = Y.shape[0]
        pred_result = np.zeros(total)
        prob = self.prob * 0.01
        pred_result[self.threhold > prob] = 1
        self.precision = precision_score(Y, pred_result)
        self.recall = recall_score(Y, pred_result)
        self.f1 = f1_score(Y, pred_result)
        self.report = classification_report(Y, pred_result)
        if filename and X is not None:
            fp_file = os.path.join(self.output, "FP_" + filename)
            fp = X[(pred_result == 1) & (Y == 0)]
            self.fp = fp.shape[0]
            fn_file = os.path.join(self.output,  "FN_" + filename)
            fn = X[(pred_result == 0) & (Y == 1)]
            self.fn = fn.shape[0]
            if save:
                fp.to_csv(fp_file)
                fn.to_csv(fn_file)
            tp = X[(pred_result == 1) & (Y == 1)]
            self.tp = tp.shape[0]
            tn = X[(pred_result == 0) & (Y == 0)]
            self.tn = tn.shape[0]
            if printing:
                print("There are " + str(self.fp) + " false positive samples.")
                print("There are " + str(self.tp) + " true positive samples.")
                print("There are " + str(self.fn) + " false negative samples.")
                print("There are " + str(self.tn) + " true negative samples.")

    def save_threshold(self, threshold, filename=None):
        """
        Show the result for training and test data for each threshold
        :param threshold: The threshold value
        :param filename: The general filename
        """
        self.train_individual(self.train_X, self.train_Y, fit=False)
        self.modify_threshold(threshold, self.train_Y, "Train_" + filename, self.train_X)
        self.show_result()
        self.train_individual(self.test_X, self.test_Y, fit=False)
        self.modify_threshold(threshold, self.test_Y, "Test_" + filename, self.test_X)
        self.show_result()

    def loop_threshold(self, fileName=None, save=False,
                       thresholds=[99, 99.9, 99.91, 99.92, 99.93, 99.94, 99.95, 99.96, 99.97, 99.98, 99.99]):
        """
         Save the output with different threshold values
        :param fileName: The file name to save
        :param thresholds: A list of threshold values
        """
        self.train_individual(self.train_X, self.train_Y, fit=True)
        self.output_threshold(self.train_X, self.train_Y, "train_" + fileName, thresholds, save=save)
        self.train_individual(self.test_X, self.test_Y, fit=False)
        self.output_threshold(self.test_X, self.test_Y, "test_" + fileName, thresholds,save=save)

    def output_threshold(self, X, Y, fileName,thresholds,save=False):
        """
        Do the actual looping work
        :param X: The predictor variables
        :param Y: The response variables
        :param fileName: The output file name
        :param thresholds: The list of threshold values
        :return:
        """
        precision_list = []
        recall_list = []
        f1_list = []
        prob_list = []
        tp_list = []
        fp_list = []
        fn_list = []
        tn_list = []
        for i in thresholds:
            prefix = "".join(str(i).split("."))
            file = prefix + "_" + fileName
            self.modify_threshold(i, Y, filename=file, X=X, save=save)
            precision_list.append(self.precision)
            recall_list.append(self.recall)
            f1_list.append(self.f1)
            tp_list.append(self.tp)
            fp_list.append(self.fp)
            fn_list.append(self.fn)
            tn_list.append(self.tn)
            prob_list.append(i)
        result = pd.DataFrame({'precision': precision_list, 'recall': recall_list,
                               'f1 score': f1_list, 'true positive': tp_list,
                               'false positive': fp_list, 'true negative': tn_list,
                               'false negative': fn_list, 'probability (%)': prob_list})
        if fileName:
            new_file = self.output + fileName
            result.to_csv(new_file)
        else:
            print("No data has been saved.")

    def plot_validation_curve(self, param_range, param_name, score_metrics,
                              estimator=LogisticRegression(random_state=1, class_weight='balanced'),
                              splits=5, verbose=150, ratio=None, random_number=None, n_jobs=-1, figsize=(30,30)):
        """
        Plot the validation curve with the given training data
        :param param_range: The hyper-parameter range
        :param param_name: The hyper-parameter name
        :param score_metrics: The performance metrics to plot
        :param estimator: The predictive model
        :param splits: The number of cross validation split
        :param verbose: The verbose level
        :param ratio: The percentage of the sampling of the original dataset
        :param random_number: random number to set
        :param n_jobs: The number of parallel jobs to run
        :param figsize: size of the figure to plot
        """
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
        pipe = Pipeline([('standardlized', StandardScaler()),
                         ('classify', estimator)])
        if ratio is not None and ratio > 0:
            data1 = self.train_X[self.train_Y == 1].sample(frac=ratio, random_state=random_number)
            data2 = self.train_X[self.train_Y == 0].sample(frac=ratio, random_state=random_number)
            train_Y = pd.Series([1] * data1.shape[0] + [0] * data2.shape[0])
            train_X = pd.concat([data1, data2])
            print("For training dataset: ")
            print(train_X.shape)
            print("The distribution of the training set: ")
            print(train_Y.value_counts())
        else:
            train_Y = self.train_Y
            train_X = self.train_X
        train_scores, test_scores = validation_curve(pipe, train_X, train_Y,
                                                     param_name=param_name, cv=cv,
                                                     scoring=make_scorer(score_metrics), param_range=param_range,
                                                     verbose=verbose, n_jobs=n_jobs)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.figure(figsize=figsize)
        plt.title('Validation curve for L1 Logistic Regression')
        plt.xlabel(param_name)
        plt.ylabel(score_metrics.__name__)
        train_upper, train_lower = train_scores_mean + train_scores_std, train_scores_mean - train_scores_std
        test_upper, test_lower = test_scores_mean + test_scores_std, test_scores_mean - test_scores_std
        upper = max(np.amax(train_upper), np.amax(test_upper))
        lower = min(np.amin(train_lower), np.amin(test_lower))
        plt.ylim(lower - 0.1, upper + 0.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training Score",
                     color='darkorange', lw=lw)
        plt.fill_between(param_range, train_lower,
                         train_upper, alpha=0.2,
                         color='darkorange', lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color='navy', lw=lw)
        plt.fill_between(param_range, test_lower,
                         test_upper, alpha=0.2,
                         color='navy', lw=lw)
        plt.legend(loc='best')
        plt.show()

    def plot_learn_curve(self, estimator, splits=5, ylimit=(0.7, 1.01), numjobs=-1):
        """
        Plot the learning curve with different sample size
        :param estimator: The learning model
        :param splits: The number of splits for cross validation
        :param ylimit: limit of the Y axis
        :param numjobs: The number of parallel jobs
        """
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
        plot_learning_curve(estimator, 'Learning Curve', self.train_X, self.train_Y, ylimit, cv=cv, n_jobs=numjobs)
        plt.show()
