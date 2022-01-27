"""
Utility functions used in the random search training.
"""

from sklearn.model_selection import ParameterSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN

def under_over_sampler(X, y, method=None, ratio=0.5):
    """Returns an undersampled or oversampled data set. Implemented using imbalanced-learn package.
    ['random_over','random_under','random_under_bootstrap','smote', 'adasyn']
    
    """

    if method == None:
        return X, y

    # oversample methods: https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
    elif method == "random_over":
        # print('before:',sorted(Counter(y).items()))
        ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        # print('after:',sorted(Counter(y_resampled).items()))
        return X_resampled, y_resampled

    elif method == "random_under":
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled

    elif method == "random_under_bootstrap":
        rus = RandomUnderSampler(
            sampling_strategy=ratio, random_state=0, replacement=True
        )
        X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled

    elif method == "smote":
        X_resampled, y_resampled = SMOTE(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(X, y)
        return X_resampled, y_resampled

    elif method == "adasyn":
        X_resampled, y_resampled = ADASYN(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(X, y)
        return X_resampled, y_resampled

    else:
        return X, y

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