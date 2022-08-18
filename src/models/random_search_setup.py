"""
Set the parameters that random search will look over for each classifier.

Default values, as provided by sklearn or XGBoost, are noted.

This script will be saved in the model record folder.
"""

from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import numpy as np
import random

# from src.models.classifiers import (
#     rf_classifier,
#     xgb_classifier,
#     knn_classifier,
#     lr_classifier,
#     sgd_classifier,
#     ridge_classifier,
#     svm_classifier,
#     nb_classifier,
# )


###############################################################################
# General random search parameters
###############################################################################

general_params = {
    "scaler_method": [
        "standard", 
        "minmax", 
        None
        ],
    "oversamp_method": [
        "random_over",
        "smote_enn",
        "smote_tomek",
        "borderline_smote",
        "kmeans_smote",
        "svm_smote",
        "smote",
        "adasyn",
        None,
    ],
    "undersamp_method": [
        "random_under",
        "random_under_bootstrap",
        None,
    ],
    "oversamp_ratio": [
        0.1, 
        0.2, 
        0.3, 
        0.4, 
        0.5, 
        0.6, 
        0.7, 
        0.8, 
        0.85, 
        0.9, 
        0.95, 
        1.0
    ],
    "undersamp_ratio": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "early_stopping_rounds": [
        None, 
        10, 
        50, 
        100
    ],
    "feat_select_method": [
        # "tsfresh",
        "tsfresh_random",
        "random",
        # None,
    ],
    "max_feats": [
        # None,
        10,
        # 20,
        # 50,
        # 75,
        # 100,
        # 200,
        # 300,
        # 400,
        # 500,
    ],
    "classifier": [
        "rf",  # Random Forest
        "xgb", # XGBoost
        "knn", # K-Nearest Neighbors
        # "svm", # Support Vector Machine
        # "lr", # Logistic Regression
        # "sgd", # Linear model with SGD
        # "ridge", # Ridge Classifier
        # "nb",  # Naive Bayes
    ],
    "feat_col_list": [
        None,
        # ['current__fft_coefficient__attr_"angle"__coeff_39', 'current__fft_coefficient__attr_"imag"__coeff_26', 'current__fft_coefficient__attr_"angle"__coeff_26', 'current__fft_coefficient__attr_"real"__coeff_42', 'current__fft_coefficient__attr_"imag"__coeff_36', 'current__fft_coefficient__attr_"abs"__coeff_33', 'current__fft_coefficient__attr_"abs"__coeff_2', 'current__fft_coefficient__attr_"abs"__coeff_19', 'current__fft_coefficient__attr_"angle"__coeff_13', 'current__cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)']
    ],
    "dataprep_method": [
        "cnc_standard",
        "cnc_standard_index_select",
        # "cnc_index_transposed",
        # "cnc_index_select_transposed",
    ],
    "cnc_indices_keep": [
        list(range(0, 10)),
        list(range(1, 10)),
        list(range(1, 9)),
        list(range(1, 8)),
        list(range(2, 8)),
        list(range(2, 9)),
        list(range(2, 10)),
    ],  # no cut indices past 9 that are valid
    "cnc_cases_drop": [
        None,  # no drop
        # True,  # randomly select cases to drop
        # [1,2], # input a list of cases to drop
        # [29, 30, 31, 33], # these are cases that have been down sampled from 2khz
        # [9, 17, 21, 23, 25, 31, 35, 15],
        # [9],
        # [17],
        # [21],
        # [23],
        # [25],
        # [31],
        # [35],
        # [15],
        # [9, 17],
        # [9, 21],
        # [9, 23],
        # [9, 25],
        # [9, 31],
        # [9, 35],
        # [9, 15],
        # [17, 21],
        # [17, 23],
        # [17, 25],
        # [17, 31],
        # [17, 35],
        # [17, 15],
        # [21, 23],
        # [21, 25],
        # [21, 31],
        # [21, 35],
        # [21, 15],
        # [23, 25],
        # [23, 31],
        # [23, 35],
        # [23, 15],
        # [25, 31],
        # [25, 35],
        # [25, 15],
        # [31, 35],
        # [31, 15],
        # [35, 15],
    ],
}


###############################################################################
# Classifier parameters
###############################################################################

# random forest parameters
rf_params = {
    "n_estimators": sp_randint(5, 500),  # default=100
    "criterion": ["gini", "entropy"],  # default=gini
    "max_depth": sp_randint(1, 500),  # default=None
    "min_samples_leaf": sp_randint(1, 10),
    "bootstrap": [True, False],
    "min_samples_split": sp_randint(2, 10),  # default=2
    "class_weight": ["balanced", "balanced_subsample", None],
}

# xgboost parameters
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
xgb_params = {
    "max_depth": sp_randint(1, 64),
    "eta": [0.1, 0.3, 0.7],
    "objective": ["binary:logistic"],
    "eval_metric": ["error", "aucpr"],
    # "early_stopping_rounds": [None, 10, 20, 50, 100],
    "seed": sp_randint(1, 2 ** 16),
    "scale_pos_weight": sp_randint(1, 100),
    "lambda": [0.0, 0.5, 1, 1.5, 3],
    "alpha": [0, 0.5, 1, 1.5, 3],
}

# knn parameters
knn_params = {
    "n_neighbors": sp_randint(1, 25),
    "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
    "weights": ["uniform", "distance"],
}

# logistic regression parameters
lr_params = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "solver": ["saga", "lbfgs"],
    "class_weight": [None, "balanced"],
    "l1_ratio": uniform(loc=0, scale=1),
    "max_iter": [20000],
}

# sgd parameters
sgd_params = {
    "penalty": [
        "l1",
        "l2",
        "elasticnet",
    ],
    "loss": [
        "hinge",
        "log",
        "modified_huber",
        "squared_hinge",
        "perceptron",
        "squared_error",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ],
    "fit_intercept": [True, False],
    "l1_ratio": uniform(loc=0, scale=1),
    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "eta0": uniform(loc=0, scale=2),
}

# ridge parameters
ridge_params = {
    "alpha": uniform(loc=0, scale=5),
    # "normalize": [True, False],
}

# svm parameters
svm_params = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": sp_randint(2, 5),
    "C": np.round(
        np.arange(0.01, 3.0, 0.02), 2
    ),  # uniform(loc=0, scale=2), # [0.01, 0.1, 0.5, 0.7, 1.0, 1.3, 2.0, 5.0],
    "max_iter": [25000],
    "verbose": [False],
}

# gaussian parameters
nb_params = {}
