"""
Utility functions used in the random search training.
"""

from sklearn.model_selection import ParameterSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

from src.models.random_search_setup import (
    rf_params,
    xgb_params,
    knn_params,
    lr_params,
    sgd_params,
    ridge_params,
    svm_params,
    nb_params,
)

from src.models.classifiers import (
    rf_classifier,
    xgb_classifier,
    knn_classifier,
    lr_classifier,
    sgd_classifier,
    ridge_classifier,
    svm_classifier,
    nb_classifier,
)

from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

def under_over_sampler(x, y, method=None, ratio=0.5):
    """
    Returns an undersampled or oversampled data set. Implemented using imbalanced-learn package.
    ['random_over','random_under','random_under_bootstrap','smote', 'adasyn']
    """

    if method == None:
        return x, y

    # oversample methods: https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
    elif method == "random_over":
        # print('before:',sorted(Counter(y).items()))
        ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
        x_resampled, y_resampled = ros.fit_resample(x, y)
        # print('after:',sorted(Counter(y_resampled).items()))
        return x_resampled, y_resampled

    elif method == "random_under":
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
        x_resampled, y_resampled = rus.fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "random_under_bootstrap":
        rus = RandomUnderSampler(
            sampling_strategy=ratio, random_state=0, replacement=True
        )
        x_resampled, y_resampled = rus.fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "smote":
        x_resampled, y_resampled = SMOTE(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "adasyn":
        x_resampled, y_resampled = ADASYN(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    else:
        return x, y


def get_classifier_and_params(classifier_string):
    if classifier_string == "rf":
        return rf_classifier, rf_params

    elif classifier_string == "xgb":
        return xgb_classifier, xgb_params

    elif classifier_string == "knn":
        return knn_classifier, knn_params

    elif classifier_string == "lr":
        return lr_classifier, lr_params

    elif classifier_string == "sgd":
        return sgd_classifier, sgd_params

    elif classifier_string == "ridge":
        return ridge_classifier, ridge_params

    elif classifier_string == "svm":
        return svm_classifier, svm_params

    elif classifier_string == "nb":
        return nb_classifier, nb_params

    else:
        raise ValueError("Classifier string not recognized")


def calculate_scores(clf, x_test, y_test,):
    """Helper function for calculating a bunch of scores"""

    y_pred = clf.predict(x_test)

    # need decision function or probability
    # should probably remove the try-except at a later date
    try:
        y_scores = clf.decision_function(x_test)
    except:
        y_scores = clf.predict_proba(x_test)[:, 1]

    n_thresholds = len(np.unique(y_scores))

    n_correct = sum(y_pred == y_test)

    # need to use decision scores, or probabilities, in roc_score
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_scores)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)

    # calculate the precision recall curve and roc_auc curve
    # when to use ROC vs. precision-recall curves, Jason Brownlee http://bit.ly/38vEgnW
    # https://stats.stackexchange.com/questions/113326/what-is-a-good-auc-for-a-precision-recall-curve
    prauc_result = auc(recalls, precisions)
    rocauc_result = roc_auc_score(y_test, y_scores)

    # calculate precision, recall, f1 scores
    precision_result = precision_score(y_test, y_pred)
    recall_result = recall_score(y_test, y_pred)
    f1_result = f1_score(y_test, y_pred)

    # create a dictionary of all the scores
    scores = {"n_correct": n_correct, "n_thresholds": n_thresholds, "prauc_result": prauc_result, "rocauc_result": rocauc_result,
                "precision_result": precision_result, "recall_result": recall_result, "f1_result": f1_result, 
                "precisions": precisions, "recalls": recalls, "pr_thresholds": pr_thresholds,
                "fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thresholds, "y_scores": y_scores}


    return scores


def scale_data(x_train, x_test, scaler_method=None):
    if scaler_method == "standard":
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    elif scaler_method == "min_max":
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    else:
        pass
    return x_train, x_test
    


###############################################################################
# Milling data functions
###############################################################################

def milling_add_y_label_anomaly(df_feat):
    """
    Adds a y label to the features dataframe and setup
    dataframe for use on milling_select_features function.

    Label schema:

    y = 0 if the tool is healthy (new-ish) or degraded
    y =1 if the tool is worn out (failed) (an anomaly)

    """
    # set up the y label
    df_feat["y"] = df_feat["tool_class"] > 1
    df_feat["y"] = df_feat["y"].astype(int)

    df_feat = df_feat.reset_index(drop=True)  # reset index just in case

    return df_feat