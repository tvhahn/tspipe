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
import logging
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from src.models.utils import (
    milling_add_y_label_anomaly,
    under_over_sampler,
    scale_data,
    calculate_scores,
    get_classifier_and_params,
    get_model_metrics_df,
)
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


def kfold_cv(
    df,
    clf,
    uo_method,
    scaler_method,
    imbalance_ratio,
    meta_label_cols,
    stratification_grouping_col=None,
    y_label_col="y",
    n_splits=5,
):

    n_thresholds_list = []
    precisions_list = []
    recalls_list = []
    precision_score_list = []
    recall_score_list = []
    fpr_list = []
    tpr_list = []
    prauc_list = []
    rocauc_list = []
    f1_list = []
    accuracy_list = []

    # perform stratified k-fold cross validation using the grouping of the y-label and another column
    if (
        stratification_grouping_col is not None
        and stratification_grouping_col is not y_label_col
    ):
        df_strat = df[[stratification_grouping_col, y_label_col]].drop_duplicates()

        skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # use clone to do a deep copy of model without copying attached data
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        clone_clf = clone(clf)

        for train_index, test_index in skfolds.split(
            df_strat[[stratification_grouping_col]], df_strat[["y"]]
        ):
            train_strat_vals = df_strat.iloc[train_index][
                stratification_grouping_col
            ].values
            test_strat_vals = df_strat.iloc[test_index][
                stratification_grouping_col
            ].values

            x_train = df[df[stratification_grouping_col].isin(train_strat_vals)]
            y_train = x_train[y_label_col].values.astype(int)
            x_train = x_train.drop(meta_label_cols + [y_label_col], axis=1).values

            x_test = df[df[stratification_grouping_col].isin(train_strat_vals)]
            y_test = x_test[y_label_col].values.astype(int)
            x_test = x_test.drop(meta_label_cols + [y_label_col], axis=1).values

            # scale the data
            scale_data(x_train, x_test, scaler_method)

            # under-over-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=uo_method, ratio=imbalance_ratio
            )

            # train model
            clone_clf.fit(x_train, y_train)

            # calculate the scores for each individual model train in the cross validation
            # save as a dictionary: "ind_score_dict"
            ind_score_dict = calculate_scores(clone_clf, x_test, y_test)

            n_thresholds_list.append(ind_score_dict["n_thresholds"])
            precisions_list.append(ind_score_dict["precisions"])
            recalls_list.append(ind_score_dict["recalls"])
            precision_score_list.append(ind_score_dict["precision_result"])
            recall_score_list.append(ind_score_dict["recall_result"])
            fpr_list.append(ind_score_dict["fpr"])
            tpr_list.append(ind_score_dict["tpr"])
            prauc_list.append(ind_score_dict["prauc_result"])
            rocauc_list.append(ind_score_dict["rocauc_result"])
            f1_list.append(ind_score_dict["f1_result"])
            accuracy_list.append(ind_score_dict["accuracy_result"])

    # perform stratified k-fold cross if only using the y-label for stratification
    else:
        skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # use clone to do a deep copy of model without copying attached data
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        clone_clf = clone(clf)

        for train_index, test_index in skfolds.split(df, df[["y"]]):
            df_train = df.iloc[train_index]
            df_test = df.iloc[test_index]

            y_train = df_train[y_label_col].values.astype(int)
            x_train = df_train.drop(meta_label_cols + [y_label_col], axis=1).values

            y_test = df_test[y_label_col].values.astype(int)
            x_test = df_test.drop(meta_label_cols + [y_label_col], axis=1).values

            # scale the data
            scale_data(x_train, x_test, scaler_method)

            # under-over-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=uo_method, ratio=imbalance_ratio
            )

            # train model
            clone_clf.fit(x_train, y_train)

            # calculate the scores for each individual model train in the cross validation
            # save as a dictionary: "ind_score_dict"
            ind_score_dict = calculate_scores(clone_clf, x_test, y_test)

            n_thresholds_list.append(ind_score_dict["n_thresholds"])
            precisions_list.append(ind_score_dict["precisions"])
            recalls_list.append(ind_score_dict["recalls"])
            precision_score_list.append(ind_score_dict["precision_result"])
            recall_score_list.append(ind_score_dict["recall_result"])
            fpr_list.append(ind_score_dict["fpr"])
            tpr_list.append(ind_score_dict["tpr"])
            prauc_list.append(ind_score_dict["prauc_result"])
            rocauc_list.append(ind_score_dict["rocauc_result"])
            f1_list.append(ind_score_dict["f1_result"])
            accuracy_list.append(ind_score_dict["accuracy_result"])

    n_thresholds_array = np.array(n_thresholds_list, dtype=int)
    precisions_array = np.array(precisions_list, dtype=object)
    recalls_array = np.array(recalls_list, dtype=object)
    precision_score_array = np.array(precision_score_list, dtype=object)
    recall_score_array = np.array(recall_score_list, dtype=object)
    fpr_array = np.array(fpr_list, dtype=object)
    tpr_array = np.array(tpr_list, dtype=object)
    prauc_array = np.array(prauc_list, dtype=object)
    rocauc_array = np.array(rocauc_list, dtype=object)
    f1_score_array = np.array(f1_list, dtype=object)
    accuracy_array = np.array(accuracy_list, dtype=object)

    # create a dictionary of the result arrays
    trained_result_dict = {
        "precisions_array": precisions_array,
        "recalls_array": recalls_array,
        "precision_score_array": precision_score_array,
        "recall_score_array": recall_score_array,
        "fpr_array": fpr_array,
        "tpr_array": tpr_array,
        "prauc_array": prauc_array,
        "rocauc_array": rocauc_array,
        "f1_score_array": f1_score_array,
        "n_thresholds_array": n_thresholds_array,
        "accuracy_array": accuracy_array,
    }

    return trained_result_dict


def train_single_model(
    df, sampler_seed, meta_label_cols, stratification_grouping_col=None, y_label_col="y"
):
    # generate the list of parameters to sample over
    params_dict_train_setup = list(
        ParameterSampler(general_params, n_iter=1, random_state=sampler_seed)
    )[0]

    uo_method = params_dict_train_setup["uo_method"]
    scaler_method = params_dict_train_setup["scaler_method"]
    imbalance_ratio = params_dict_train_setup["imbalance_ratio"]
    classifier = params_dict_train_setup["classifier"]
    print(
        f"classifier: {classifier}, uo_method: {uo_method}, imbalance_ratio: {imbalance_ratio}"
    )

    # get classifier and its parameters
    clf_function, params_clf = get_classifier_and_params(classifier)

    # instantiate the model
    clf, param_dict_clf_raw, params_dict_clf_named = clf_function(
        sampler_seed, params_clf
    )
    print("\n", params_dict_clf_named)

    model_metrics_dict = kfold_cv(
        df,
        clf,
        uo_method,
        scaler_method,
        imbalance_ratio,
        meta_label_cols,
        stratification_grouping_col,
        y_label_col,
    )

    # added additional parameters to the training setup dictionary
    params_dict_train_setup["sampler_seed"] = sampler_seed

    return model_metrics_dict, params_dict_clf_named, params_dict_train_setup


def random_search_runner(
    df,
    rand_search_iter,
    meta_label_cols,
    stratification_grouping_col,
    path_save_dir=None,
    y_label_col="y",
    save_freq=1,
    debug=False,
):

    df_results = pd.DataFrame()
    for i in range(rand_search_iter):
        # set random sample seed
        sample_seed = random.randint(0, 2 ** 25)
        sample_seed = 10


        if i == 0:
            file_name = f"results_{sample_seed}.csv"
        
        try:

            (
                model_metrics_dict,
                params_dict_clf_named,
                params_dict_train_setup,
            ) = train_single_model(
                df,
                sample_seed,
                meta_label_cols,
                stratification_grouping_col,
                y_label_col,
            )

            df_t = pd.DataFrame.from_dict(
                params_dict_train_setup, orient="index"
            ).T  # train setup params
            df_c = pd.DataFrame.from_dict(
                params_dict_clf_named, orient="index"
            ).T  # classifier params
            df_m = get_model_metrics_df(model_metrics_dict)

            df_results = df_results.append(pd.concat([df_t, df_m, df_c], axis=1))

            if i % save_freq == 0:
                if path_save_dir is not None:
                    df_results.to_csv(path_save_dir / file_name, index=False)
                else:
                    df_results.to_csv(file_name, index=False)
        # except Exception as e and log the exception
        except Exception as e:
            if debug:
                print('####### Exception #######')
                print(e)
                logging.exception(f"##### Exception in random_search_runner:\n{e}\n\n")
            pass


def main():

    root_path = Path().cwd()
    print('root_path: ', root_path)
    path_data_folder = Path().cwd() / "data"
    print(path_data_folder)

    folder_raw_data_milling = path_data_folder / "raw/milling"
    folder_interim_data_milling = path_data_folder / "interim/milling"
    folder_processed_data_milling = path_data_folder / "processed/milling"
    folder_models = root_path / "models"

    Y_LABEL_COL = "y"

    # identify if there is another column you want to
    # stratify on, besides the y label
    STRATIFICATION_GROUPING_COL = "cut_no"

    # list of the columns that are not features columns
    # (not including the y-label column)
    META_LABEL_COLS = ["cut_id", "cut_no", "case", "tool_class"]

    RAND_SEARCH_ITER = 2

    # set a seed for the parameter sampler
    # SAMPLER_SEED = random.randint(0, 2 ** 16)

    # load feature dataframe
    df = pd.read_csv(
        folder_processed_data_milling / "milling.csv",
    )

    # add y label
    df = milling_add_y_label_anomaly(df)

    Y_LABEL_COL = "y"

    # identify if there is another column you want to
    # stratify on, besides the y label
    STRATIFICATION_GROUPING_COL = "cut_no"

    # list of the columns that are not features columns
    # (not including the y-label column)
    META_LABEL_COLS = ["cut_id", "cut_no", "case", "tool_class"]

    LOG_FILENAME = folder_models / 'logging_example.out'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    random_search_runner(
        df,
        RAND_SEARCH_ITER,
        META_LABEL_COLS,
        STRATIFICATION_GROUPING_COL,
        path_save_dir= root_path / "models",
        y_label_col="y",
        save_freq=1,
        debug=True,
    )

if __name__ == "__main__":
    main()