import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import re

# sklearn classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
# from src.models.random_search_setup import (
#     rf_params,
#     xgb_params,
#     knn_params,
#     lr_params,
#     sgd_params,
#     ridge_params,
#     svm_params,
#     gaussian_params,
# )


def get_param_dict_named(clf, param_dict_raw):

    name = re.sub("'", "", str(type(clf)).replace(">", "").split(".")[-1])

    # rebuild the parameter dict and append the classifier name onto each parameter
    param_dict_named = {}
    for k in param_dict_raw:
        param_dict_named[str(name) + "_" + k] = param_dict_raw[k]

    return param_dict_named


# random forest
def rf_classifier(sampler_seed, params):

    param_dict_raw = list(
        ParameterSampler(params, n_iter=1, random_state=sampler_seed)
    )[0]

    clf = RandomForestClassifier(random_state=sampler_seed, **param_dict_raw)

    return (
        clf,
        param_dict_raw,
        get_param_dict_named(clf, param_dict_raw),
    )


# XGBoost
def xgb_classifier(sampler_seed, params):

    param_dict_raw = list(
        ParameterSampler(params, n_iter=1, random_state=sampler_seed)
    )[0]

    # rebuild the parameter dict and append the classifier name onto each parameter
    param_dict_named = {}
    for k in param_dict_raw:
        param_dict_named["XGB" + "_" + k] = param_dict_raw[k]

    clf = XGBClassifier(random_state=sampler_seed, **param_dict_raw)

    return clf, param_dict_raw, param_dict_named


# knn
def knn_classifier(sampler_seed, params):
    parameters_sample_dict = {
        "n_neighbors": sp_randint(1, 25),
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "weights": ["uniform", "distance"],
    }

    param_dict_raw = list(
        ParameterSampler(params, n_iter=1, random_state=sampler_seed)
    )[0]

    clf = KNeighborsClassifier(**param_dict_raw)

    return (
        clf,
        param_dict_raw,
        get_param_dict_named(clf, param_dict_raw),
    )


# logistic regression
def lr_classifier(sampler_seed, params):

    param_dict_raw = list(
        ParameterSampler(params, n_iter=1, random_state=sampler_seed)
    )[0]

    # l1_ratio is only used with elasticnet penalty
    if param_dict_raw["penalty"] != "elasticnet":
        param_dict_raw["l1_ratio"] = None
    if param_dict_raw["penalty"] in ["l1", "elasticnet"]:
        param_dict_raw["solver"] = "saga"

    clf = LogisticRegression(random_state=sampler_seed, **param_dict_raw)

    return (
        clf,
        param_dict_raw,
        get_param_dict_named(clf, param_dict_raw),
    )


# stochastic gradient descent classifier
def sgd_classifier(sampler_seed, params):

    param_dict_raw = list(
        ParameterSampler(params, n_iter=1, random_state=sampler_seed)
    )[0]

    # add any constrains to the parameter dict
    if param_dict_raw["learning_rate"] == "optimal":
        param_dict_raw["eta0"] = 0.0

    clf = SGDClassifier(random_state=sampler_seed, **param_dict_raw)

    return (
        clf,
        param_dict_raw,
        get_param_dict_named(clf, param_dict_raw),
    )


# ridge
def ridge_classifier(sampler_seed, params):

    param_dict_raw = list(
        ParameterSampler(params, n_iter=1, random_state=sampler_seed)
    )[0]

    clf = RidgeClassifier(random_state=sampler_seed, **param_dict_raw)

    return (
        clf,
        param_dict_raw,
        get_param_dict_named(clf, param_dict_raw),
    )


# svm
def svm_classifier(sampler_seed, params):

    param_dict_raw = list(
        ParameterSampler(params, n_iter=1, random_state=sampler_seed)
    )[0]

    # add any constrains to the parameter dict
    if param_dict_raw["kernel"] != "poly":
        param_dict_raw["degree"] = None

    clf = SVC(random_state=sampler_seed, **param_dict_raw)

    return (
        clf,
        param_dict_raw,
        get_param_dict_named(clf, param_dict_raw),
    )


# gaussian naive bayes
def nb_classifier(sampler_seed, params):

    param_dict_raw = list(
        ParameterSampler(params, n_iter=1, random_state=sampler_seed)
    )[0]

    clf = GaussianNB(**param_dict_raw)

    return (
        clf,
        param_dict_raw,
        get_param_dict_named(clf, param_dict_raw),
    )
