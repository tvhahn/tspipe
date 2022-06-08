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
        # "standard", 
        # "minmax",
        None
    ],
    "uo_method": [
        # "random_over",
        # "random_under",
        # "random_under_bootstrap",
        # "smote",
        # "adasyn",
        None,
    ],
    "imbalance_ratio": [0.1, 0.3, 0.5, 0.7, 0.8, 1.0],
    "classifier": [
        "rf",
        "xgb",
        "knn",
        "lr",
        "sgd",
        "ridge",
        "svm",
        "nb",
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
    "max_depth": sp_randint(2, 64),
    "eta": [0.1, 0.3, 0.7],
    "objective": ["binary:logistic"],
    "eval_metric": ["error", "aucpr"],
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
        "squared_loss",
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
    "C":  np.round(np.arange(0.01, 3.0, 0.02), 2), # uniform(loc=0, scale=2), # [0.01, 0.1, 0.5, 0.7, 1.0, 1.3, 2.0, 5.0],
    "max_iter": [25000],
    "verbose": [False],
}

# gaussian parameters
nb_params = {}
