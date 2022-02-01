import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import seaborn as sns
import re
import random
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from src.models.utils import milling_add_y_label_anomaly, under_over_sampler
from src.models.random_search_setup import general_params
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

from src.visualization.visualize import plot_pr_roc_curves_kfolds