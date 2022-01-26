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

def build_classifier_parameter_dict(clf, p_clf):

    name = re.sub("'", "", str(type(clf)).replace(">", "").split(".")[-1])

    # rebuild the parameter dict and append the classifier name onto each parameter
    classifier_parameters = {}
    for k in p_clf:
        classifier_parameters[str(name) + "_" + k] = p_clf[k]

    return classifier_parameters

def random_forest_classifier(parameter_sampler_random_int):
    parameters_sample_dict = {
        "n_estimators": sp_randint(5, 500),
        "max_depth": sp_randint(1, 500),
        "random_state": sp_randint(1, 2 ** 16),
        "min_samples_leaf": sp_randint(1, 10),
        "bootstrap": [True, False],
        "class_weight": ['balanced','balanced_subsample',None],
    }

    p_clf = list(
        ParameterSampler(
            parameters_sample_dict, n_iter=1, random_state=parameter_sampler_random_int
        )
    )[0]

    clf = RandomForestClassifier(**p_clf)

    return clf, build_classifier_parameter_dict(clf, p_clf)


def xgboost_classifier(parameter_sampler_random_int):
    '''https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn'''
    parameters_sample_dict = {
        "max_depth": sp_randint(2, 64),
        "eta": [0.1,0.3,0.7],
        "objective": ['binary:logistic'],
        "eval_metric": ['error','aucpr'],
        "seed": sp_randint(1, 2 ** 16),
        "scale_pos_weight": sp_randint(1, 100),
        "lambda": [0.0, 0.5, 1, 1.5, 3],
        "alpha": [0, 0.5, 1, 1.5 ,3]
    }


    p_clf = list(
        ParameterSampler(
            parameters_sample_dict, n_iter=1, random_state=parameter_sampler_random_int
        )
    )[0]


    # rebuild the parameter dict and append the classifier name onto each parameter
    classifier_parameters = {}
    for k in p_clf:
        classifier_parameters['XGB' + "_" + k] = p_clf[k]

    clf = XGBClassifier(**p_clf)

    return clf, classifier_parameters

def knn_classifier(parameter_sampler_random_int):
    parameters_sample_dict = {
        "n_neighbors": sp_randint(1, 25),
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "weights": ["uniform", "distance"],
    }

    p_clf = list(
        ParameterSampler(
            parameters_sample_dict, n_iter=1, random_state=parameter_sampler_random_int
        )
    )[0]

    clf = KNeighborsClassifier(**p_clf)

    return clf, build_classifier_parameter_dict(clf, p_clf)


def logistic_regression(parameter_sampler_random_int):
    parameters_sample_dict = {
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "random_state": sp_randint(1, 2 ** 16),
        "solver": ["saga", "lbfgs"],
        "class_weight": [None, "balanced"],
        "l1_ratio": uniform(loc=0, scale=1),
        "max_iter": [20000]
    }

    p_clf = list(
        ParameterSampler(
            parameters_sample_dict, n_iter=1, random_state=parameter_sampler_random_int
        )
    )[0]

    # l1_ratio is only used with elasticnet penalty
    if p_clf["penalty"] != "elasticnet":
        p_clf["l1_ratio"] = None
    if p_clf["penalty"] in ["l1", "elasticnet"]:
        p_clf["solver"] = "saga"

    clf = LogisticRegression(**p_clf)
    
    return clf, build_classifier_parameter_dict(clf, p_clf)


def sgd_classifier(parameter_sampler_random_int):
    parameters_sample_dict = {
        "penalty": ["l1", "l2", "elasticnet",],
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
        "random_state": sp_randint(1, 2 ** 16),
        "fit_intercept": [True, False],
        "l1_ratio": uniform(loc=0, scale=1),
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "eta0": uniform(loc=0, scale=2),
    }

    p_clf = list(
        ParameterSampler(
            parameters_sample_dict, n_iter=1, random_state=parameter_sampler_random_int
        )
    )[0]

    if p_clf["learning_rate"] == "optimal":
        p_clf["eta0"] = 0.0

    clf = SGDClassifier(**p_clf)
    name = re.sub("'", "", str(type(clf)).replace(">", "").split(".")[-1])

    # rebuild the parameter dict and append the classifier name onto each parameter
    classifier_parameters = {}
    for k in p_clf:
        classifier_parameters[str(name) + "_" + k] = p_clf[k]

    return clf, classifier_parameters


def ridge_classifier(parameter_sampler_random_int):
    parameters_sample_dict = {
        "alpha": uniform(loc=0, scale=5),
        "random_state": sp_randint(1, 2 ** 16),
        "normalize": [True, False],
    }

    p_clf = list(
        ParameterSampler(
            parameters_sample_dict, n_iter=1, random_state=parameter_sampler_random_int
        )
    )[0]

    clf = RidgeClassifier(**p_clf)
    name = re.sub("'", "", str(type(clf)).replace(">", "").split(".")[-1])

    # rebuild the parameter dict and append the classifier name onto each parameter
    classifier_parameters = {}
    for k in p_clf:
        classifier_parameters[str(name) + "_" + k] = p_clf[k]

    return clf, classifier_parameters


def svm_classifier(parameter_sampler_random_int):
    parameters_sample_dict = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": sp_randint(1, 5),
        "C": uniform(loc=0, scale=5),
        "max_iter": [25000],
        "verbose": [False],
    }

    p_clf = list(
        ParameterSampler(
            parameters_sample_dict, n_iter=1, random_state=parameter_sampler_random_int
        )
    )[0]

    clf = SVC(**p_clf)
    name = re.sub("'", "", str(type(clf)).replace(">", "").split(".")[-1])

    if p_clf["kernel"] != "poly":
        p_clf["degree"] = None

    # rebuild the parameter dict and append the classifier name onto each parameter
    classifier_parameters = {}
    for k in p_clf:
        classifier_parameters[str(name) + "_" + k] = p_clf[k]

    return clf, classifier_parameters


def gaussian_nb_classifier(parameter_sampler_random_int):
    parameters_sample_dict = {}

    p_clf = list(
        ParameterSampler(
            parameters_sample_dict, n_iter=1, random_state=parameter_sampler_random_int
        )
    )[0]

    clf = GaussianNB(**p_clf)
    name = re.sub("'", "", str(type(clf)).replace(">", "").split(".")[-1])

    # rebuild the parameter dict and append the classifier name onto each parameter
    classifier_parameters = {}
    for k in p_clf:
        classifier_parameters[str(name) + "_" + k] = p_clf[k]

    return clf, classifier_parameters

