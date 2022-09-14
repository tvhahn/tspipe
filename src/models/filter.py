import pandas as pd
from pathlib import Path
import argparse
import logging
import os
import matplotlib
import numpy as np
import pickle

# run matplotlib without display
# https://stackoverflow.com/a/4706614/9214620
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models.train import train_single_model, load_cnc_features
from src.models.utils import (
    milling_add_y_label_anomaly,
    cnc_add_y_label_binary,
    get_model_metrics_df,
    get_rates,
)
from ast import literal_eval
from src.visualization.visualize import plot_pr_roc_curves_kfolds


def set_directories(args):

    scratch_path = Path.home() / "scratch"

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    if args.path_model_dir:
        path_model_dir = Path(args.path_model_dir)
    else:
        if scratch_path.exists():
            print("Assume on HPC")
            path_model_dir = scratch_path / "feat-store" / "models"
        else:
            print("Assume on local compute")
            path_model_dir = proj_dir / "models"

    path_interim_dir = path_model_dir / args.interim_dir_name
    path_final_dir = path_model_dir / args.final_dir_name
    path_processed_dir = (
        path_data_dir / "processed" / args.dataset / args.processed_dir_name
    )

    return (
        proj_dir,
        path_data_dir,
        path_processed_dir,
        path_model_dir,
        path_interim_dir,
        path_final_dir,
    )


def filter_results_df(df, keep_top_n=None):
    dfr = df[
        (df["prauc_min"] < 1)
        & (df["prauc_max"] < 1)
        & (df["prauc_avg"] < 1)
        & (df["prauc_std"] > 0)
        & (df["rocauc_min"] < 1)
        & (df["rocauc_max"] < 1)
        & (df["rocauc_avg"] < 1)
        & (df["rocauc_std"] > 0)
        & (df["n_thresholds_min"] > 10)
        & (df["n_thresholds_max"] > 10)
        # none of the below metrics have been tuned
        # on their threshold, and therefore cannot be used
        # as a good metric for selecting the best model
        # & df["precision_score_min"] > 0)
        # & (df["precision_score_max"] < 1)
        # & (df["precision_score_std"] > 0)
        # & (df["recall_score_min"] > 0)
        # & (df["recall_score_max"] < 1)
        # & (df["recall_score_std"] > 0)
        # & (df["f1_score_min"] > 0)
        # & (df["f1_score_max"] < 1)
        # & (df["f1_score_std"] > 0)
        # & (df["accuracy_min"] < 1)
        # & (df["accuracy_max"] < 1)
        # & (df["accuracy_avg"] < 1)
        # & (df["accuracy_std"] > 0)
    ].sort_values(by=["prauc_avg", "rocauc_avg", "accuracy_avg"], ascending=False)

    if keep_top_n is not None:
        return dfr[:keep_top_n].reset_index(drop=True)
    else:
        return dfr.reset_index(drop=True)


def rebuild_params_clf(df, row_idx):
    classifier_string = df.iloc[row_idx]["classifier"]
    if classifier_string == "rf":
        prefix = "RandomForestClassifier"

    elif classifier_string == "xgb":
        prefix = "XGB"

    elif classifier_string == "knn":
        prefix = "KNeighborsClassifier"

    elif classifier_string == "lr":
        prefix = "LogisticRegression"

    elif classifier_string == "sgd":
        prefix = "SGDClassifier"

    elif classifier_string == "ridge":
        prefix = "RidgeClassifier"

    elif classifier_string == "svm":
        prefix = "SVC"

    elif classifier_string == "nb":
        prefix = "GaussianNB"

    params_clf = {
        c.replace(f"{prefix}_", ""): df.iloc[row_idx][c]
        for c in df.iloc[row_idx].dropna().index
        if c.startswith(prefix)
    }

    # convert any whole numbers in clf_cols to int
    for k in params_clf.keys():
        if isinstance(params_clf[k], float) and params_clf[k].is_integer():
            params_clf[k] = int(params_clf[k])

    return {k: [params_clf[k]] for k in params_clf.keys()}  # put each value in a list


def order_columns_on_results_df(df, dataset_name=None):

    primary_cols = [
        "classifier",
        "sampler_seed",
        "date_time",
        "dataset",
        "dataprep_method",
        "feat_file_name",
        "id",
        "meta_label_cols",
        "feat_select_method",
        "max_feats",
        "n_feats",
        "feat_col_list",
        "stratification_grouping_col",
        "y_label_col",
        "scaler_method",
        "oversamp_method",
        "oversamp_ratio",
        "undersamp_method",
        "undersamp_ratio",
        "early_stopping_rounds",
        "mcc_avg",
        "mcc_min",
        "mcc_max",
        "mcc_std",
        "prauc_avg",
        "prauc_min",
        "prauc_max",
        "prauc_std",
        "tn_fp_fn_tp_worst_prauc",
        "rocauc_avg",
        "rocauc_min",
        "rocauc_max",
        "rocauc_std",
        "accuracy_avg",
        "accuracy_min",
        "accuracy_max",
        "accuracy_std",
        "precision_score_avg",
        "precision_score_min",
        "precision_score_max",
        "precision_score_std",
        "recall_score_avg",
        "recall_score_min",
        "recall_score_max",
        "recall_score_std",
        "f1_score_avg",
        "f1_score_min",
        "f1_score_max",
        "f1_score_std",
        "n_thresholds_min",
        "n_thresholds_max",
        "test_strat_group_worst_prauc",
    ]

    if dataset_name == "cnc":
        primary_cols = [
            "classifier",
            "sampler_seed",
            "date_time",
            "dataset",
            "dataprep_method",
            "feat_file_name",
            "label_file_name",
            "id",
            "meta_label_cols",
            "feat_select_method",
            "max_feats",
            "n_feats",
            "cnc_indices_keep",
            "cnc_cases_drop",
            "feat_col_list",
            "stratification_grouping_col",
            "y_label_col",
            "scaler_method",
            "oversamp_method",
            "oversamp_ratio",
            "undersamp_method",
            "undersamp_ratio",
            "early_stopping_rounds",
            "mcc_avg",
            "mcc_min",
            "mcc_max",
            "mcc_std",
            "prauc_avg",
            "prauc_min",
            "prauc_max",
            "prauc_std",
            "tn_fp_fn_tp_worst_prauc",
            "rocauc_avg",
            "rocauc_min",
            "rocauc_max",
            "rocauc_std",
            "accuracy_avg",
            "accuracy_min",
            "accuracy_max",
            "accuracy_std",
            "precision_score_avg",
            "precision_score_min",
            "precision_score_max",
            "precision_score_std",
            "recall_score_avg",
            "recall_score_min",
            "recall_score_max",
            "recall_score_std",
            "f1_score_avg",
            "f1_score_min",
            "f1_score_max",
            "f1_score_std",
            "n_thresholds_min",
            "n_thresholds_max",
            "test_strat_group_worst_prauc",
        ]

    # remove any columns names from primary_cols that are not in df
    primary_cols = [col for col in primary_cols if col in df.columns]

    # get secondary columns, which are all the remaining columns from the df
    secondary_cols = [col for col in df.columns if col not in primary_cols]

    # return the df with the primary columns first, then the secondary columns
    return df[primary_cols + secondary_cols]


def rebuild_general_params(df, row_idx, general_param_keys=None):
    if general_param_keys is None:
        general_param_keys = [
            "scaler_method",
            "oversamp_method",
            "undersamp_method",
            "oversamp_ratio",
            "undersamp_ratio",
            "classifier",
            "early_stopping_rounds",
            "feat_select_method",
            "max_feats",
            "dataprep_method",
            "feat_col_list",
        ]

    # remove any keys from general_param_keys that are not in df
    general_param_keys = [col for col in general_param_keys if col in df.columns]

    return {k: [df.iloc[row_idx][k]] for k in general_param_keys}


def plot_generic(
    df,
    df_feat,
    save_n_figures,
    path_model_curves,
    dataset_name,
    check_feat_importance,
    save_models,
):

    n_rows = df.shape[0]

    for row_idx in range(save_n_figures):

        params_clf = rebuild_params_clf(df, row_idx)
        general_params = rebuild_general_params(df, row_idx)

        # construct additional info used during the model training
        meta_label_cols = literal_eval(df.iloc[row_idx]["meta_label_cols"])
        stratification_grouping_col = df.iloc[row_idx]["stratification_grouping_col"]
        y_label_col = df.iloc[row_idx]["y_label_col"]
        sample_seed = int(df.iloc[row_idx]["sampler_seed"])
        sample_seed_clf = sample_seed
        id = df.iloc[row_idx]["id"]
        k_folds = int(df.iloc[row_idx]["k_folds"])

        # update general_params the saved feat_col_list as in the df
        feat_col_list = literal_eval(df.iloc[row_idx]["feat_col_list"])
        general_params["feat_col_list"] = [feat_col_list]

        # rebuild parameters that are specific to certain datasets
        if dataset_name == "cnc":
            try:
                general_params["cnc_indices_keep"] = [
                    literal_eval(df.iloc[row_idx]["cnc_indices_keep"])
                ]
            except:
                general_params["cnc_indices_keep"] = [None]
            try:
                general_params["cnc_cases_drop"] = [
                    literal_eval(df.iloc[row_idx]["cnc_cases_drop"])
                ]
            except:
                general_params["cnc_cases_drop"] = [None]

        (
            model_metrics_dict,
            params_dict_clf_named,
            params_dict_train_setup,
            feat_col_list,
            meta_label_cols,
        ) = train_single_model(
            df_feat,
            sample_seed,
            sample_seed_clf,
            meta_label_cols,
            k_folds,
            stratification_grouping_col,
            y_label_col,
            feat_col_list,
            save_model=save_models,
            model_save_name=id,
            model_save_path=path_model_curves,
            general_params=general_params,
            params_clf=params_clf,
            dataset_name=dataset_name,
            check_feat_importance=check_feat_importance,
        )

        # save model_metrics_dict as a pickle file
        # model_metrics_dict_pickle_name = f"{id}_model_metrics_dict.pkl"
        # with open(path_model_curves.parent / model_metrics_dict_pickle_name, "wb") as f:
        #     pickle.dump(model_metrics_dict, f)

        i_argmin = np.argmin(model_metrics_dict["prauc_array"])
        i_argmax = np.argmax(model_metrics_dict["prauc_array"])
        print(
            "f1 score tuned, worst k-fold:",
            model_metrics_dict["f1_threshold_tuned"][i_argmin],
        )
        print(
            "confusion matrix tuned, worst k-fold:",
            model_metrics_dict["confusion_matrix_threshold_tuned"][i_argmin],
        )

        if row_idx == 0:

            # get true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp)
            # for the worst k-fold
            cm = model_metrics_dict["confusion_matrix_threshold_tuned"][i_argmin]
            tn, fp, fn, tp = cm
            tpr, fpr, fnr, tnr = get_rates(tn, fp, fn, tp)

            # create dataframe with the f1 score tuned for each k-fold
            df_tuned = pd.DataFrame(
                {
                    "id": [id],
                    "f1_threshold_tuned": [
                        model_metrics_dict["f1_threshold_tuned"][i_argmin]
                    ],
                    "threshold": [model_metrics_dict["threshold_tuned"][i_argmin]],
                    "confusion_matrix": [
                        model_metrics_dict["confusion_matrix_threshold_tuned"][i_argmin]
                    ],
                    "tpr (sensitivity)": tpr,
                    "tnr (specificity)": tnr,
                    "fpr (fall-out)": fpr,
                    "fnr (miss rate)": fnr,
                }
            )
            # save df_tuned as csv
            df_tuned.to_csv(
                path_model_curves.parent / "tuned_threshold_results.csv", index=False
            )
        else:
            cm = model_metrics_dict["confusion_matrix_threshold_tuned"][i_argmin]
            tn, fp, fn, tp = cm
            tpr, fpr, fnr, tnr = get_rates(tn, fp, fn, tp)

            df_tuned_2 = pd.DataFrame(
                {
                    "id": [id],
                    "f1_threshold_tuned": [
                        model_metrics_dict["f1_threshold_tuned"][i_argmin]
                    ],
                    "threshold": [model_metrics_dict["threshold_tuned"][i_argmin]],
                    "confusion_matrix": [
                        model_metrics_dict["confusion_matrix_threshold_tuned"][i_argmin]
                    ],
                    "tpr (sensitivity)": tpr,
                    "tnr (specificity)": tnr,
                    "fpr (fall-out)": fpr,
                    "fnr (miss rate)": fnr,
                }
            )
            df_tuned = pd.concat([df_tuned, df_tuned_2], ignore_index=True)
            df_tuned.to_csv(
                path_model_curves.parent / "tuned_threshold_results.csv", index=False
            )

        # save the feature importance df to a csv if requested
        if check_feat_importance:
            model_metrics_dict["df_feat_imp"].to_csv(
                f"{path_model_curves.parent}/{id}_feat_imp.csv", index=False
            )

        # calculate the percentage of "anomlalies" (value==1) in the df_feat. Targets are found in the "y_label_col" column.
        df_feat_anom = df_feat[df_feat[y_label_col] == 1]
        percent_anom = df_feat_anom.shape[0] / df_feat.shape[0]
        print(f"{id} has {percent_anom:.3f} anomalies")

        # print("i_argmin: ", i_argmin)
        # print("prauc_array: ", model_metrics_dict['prauc_array'])
        # print("prauc_array[i_argmin]: ", model_metrics_dict['prauc_array'][i_argmin])
        # for j in model_metrics_dict['unique_grouping']:
        #     print(j["unique_test_group"])

        plot_pr_roc_curves_kfolds(
            model_metrics_dict["precisions_array"],
            model_metrics_dict["recalls_array"],
            model_metrics_dict["fpr_array"],
            model_metrics_dict["tpr_array"],
            model_metrics_dict["rocauc_array"],
            model_metrics_dict["prauc_array"],
            percent_anomalies_truth=percent_anom,
            path_save_dir=path_model_curves,
            save_name=f"curve_{id}",
            save_plot=True,
            dpi=300,
        )

        if row_idx == n_rows - 1:
            print("No more results to save")
            break


def milling_plot_results(
    df,
    save_n_figures,
    path_dataset_processed_dir,
    feat_file_name,
    path_model_curves,
    check_feat_importance=False,
    save_models=False,
):

    # load feature dataframe
    df_feat = pd.read_csv(
        path_dataset_processed_dir / feat_file_name,
    )

    df_feat = milling_add_y_label_anomaly(df_feat)
    df_feat = df_feat.dropna(axis=1)  # drop any columns that have NaNs in them

    plot_generic(
        df,
        df_feat,
        save_n_figures,
        path_model_curves,
        dataset_name="milling",
        check_feat_importance=check_feat_importance,
        save_models=save_models,
    )


def cnc_plot_results(
    df,
    save_n_figures,
    path_dataset_processed_dir,
    feat_file_name,
    path_model_curves,
    check_feat_importance=False,
    save_models=False,
):
    # To-Do: in filter.py, add load_cnc_features to cnc_plot_results
    # df_feat = load_cnc_features(path_data_dir, path_processed_dir, feat_file_name, label_file_name)

    df_feat = pd.read_csv(
        path_dataset_processed_dir / feat_file_name,
    )
    df_feat["unix_date"] = df_feat["id"].apply(lambda x: int(x.split("_")[0]))
    df_feat["tool_no"] = df_feat["id"].apply(lambda x: int(x.split("_")[-2]))
    df_feat["index_no"] = df_feat["id"].apply(lambda x: int(x.split("_")[-1]))

    df_labels = pd.read_csv(
        path_dataset_processed_dir.parent
        / "high_level_labels_MASTER_update2022-08-18_with_case.csv"
    )
    df_feat = cnc_add_y_label_binary(df_feat, df_labels, col_list_case=["case_tool_54"])
    df_feat = df_feat.dropna(
        axis=1, how="all"
    )  # drop any columns that are completely empty
    df_feat = df_feat.dropna(axis=0)  # drop any rows that have NaN values in them

    plot_generic(
        df,
        df_feat,
        save_n_figures,
        path_model_curves,
        dataset_name="cnc",
        check_feat_importance=check_feat_importance,
        save_models=save_models,
    )


def main(args):

    (
        proj_dir,
        path_data_dir,
        path_processed_dir,
        path_model_dir,
        path_interim_dir,
        path_final_dir,
    ) = set_directories(args)

    df = pd.read_csv(
        path_final_dir / args.compiled_csv_name,
    )

    df = filter_results_df(df)
    df = order_columns_on_results_df(df, dataset_name=args.dataset)

    ####### any additional filtering
    df = df[df["n_feats"] <= 10]

    if args.dataset == "cnc":
        df = df[
            df["dataprep_method"].isin(
                ["cnc_index_select_transposed", "cnc_index_transposed"]
            )
        ]
    #     df = df[df["dataprep_method"].isin(["cnc_standard_index_select", "cnc_standard"])]

    # drop any rows where the length of the "cnc_cases_drop" column is greater than 1
    # df = df[df["cnc_cases_drop"].apply(lambda x: len(literal_eval(x)) <= 1)]

    # drop any rows where the "classifier" column is "knn" and its "f1_score" is less than 0.5
    # help from Guy on stackoverflow: https://stackoverflow.com/a/68085032/9214620
    # ids = df[(df["classifier"] == 'knn') & (df["mcc_min"] < 0.5)]["id"].values
    # df = df[~df["id"].isin(ids)]
    try:
        ids = df[
            (df["classifier"] == "rf")
            & (df["RandomForestClassifier_bootstrap"] == False)
        ]["id"].values
        df = df[~df["id"].isin(ids)]
    except:
        pass

    # df = df[(df["classifier"] == 'nb')]

    # drop any duplicate rows in df, excluding the "sampler_seed" column
    # df = df.drop_duplicates(subset=["cnc_cases_drop"], keep="first")

    # use this is you want to only select the top models by model type (e.g. top SVM, RF, etc.)
    sort_by = ["prauc_avg"]
    # sort_by = ["prauc_min"]
    # sort_by = ["prauc_min", "prauc_avg"]
    # sort_by = ["mcc_min", "mcc_avg"]
    df = (
        df.groupby(["classifier"])
        .head(int(args.keep_top_n))
        .sort_values(by=sort_by, ascending=False)
    )

    df.to_csv(path_final_dir / args.filtered_csv_name, index=False)

    if args.save_n_figures > 0:
        path_model_curves = path_final_dir / "model_curves"
        Path(path_model_curves).mkdir(parents=True, exist_ok=True)

    if args.check_feat_importance == "True":
        check_feat_importance = True
    else:
        check_feat_importance = False

    if args.save_models == "True":
        save_models = True
    else:
        save_models = False

    ########################################################################
    #### MILLING DATASET ####
    ########################################################################
    if args.dataset == "milling" and args.save_n_figures > 0:
        assert (
            df.iloc[0]["dataset"] == "milling"
        ), "dataset in results csv is not the milling dataset"

        milling_plot_results(
            df,
            args.save_n_figures,
            path_processed_dir,
            feat_file_name=args.feat_file_name,
            path_model_curves=path_model_curves,
            check_feat_importance=check_feat_importance,
            save_models=save_models,
        )

    elif args.dataset == "cnc" and args.save_n_figures > 0:

        cnc_plot_results(
            df,
            args.save_n_figures,
            path_processed_dir,
            feat_file_name=args.feat_file_name,
            path_model_curves=path_model_curves,
            check_feat_importance=check_feat_importance,
            save_models=save_models,
        )
    else:
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build data sets for analysis")

    parser.add_argument(
        "--keep_top_n",
        type=int,
        default=1,
        help="Keep the top N models in the filtered results CSV.",
    )

    parser.add_argument(
        "--save_n_figures",
        type=int,
        default=0,
        help="Keep the top N models in the filtered results CSV.",
    )

    parser.add_argument(
        "--final_dir_name",
        type=str,
        help="Folder name containing compiled csv.",
    )

    parser.add_argument(
        "--path_data_dir",
        dest="path_data_dir",
        type=str,
        help="Location of the data folder, containing the raw, interim, and processed folders",
    )

    parser.add_argument(
        "--path_model_dir",
        dest="path_model_dir",
        type=str,
        help="Folder containing the trained model results",
    )

    parser.add_argument(
        "--dataset",
        default="milling",
        type=str,
        help="Dataset used in training. Options are 'milling', 'cnc', 'airbus'",
    )

    parser.add_argument(
        "--feat_file_name",
        type=str,
        help="Name of the feature file to use. Should be found in path_data_dir/processed/{dataset}/feat_file_name",
    )

    parser.add_argument(
        "--sample_seed",
        type=int,
        help="Fix the random seed for the sampler",
    )

    parser.add_argument(
        "--sample_seed_clf",
        type=int,
        help="Fix the random seed for ONLY the classifier sampler",
    )

    parser.add_argument(
        "--compiled_csv_name",
        type=str,
        default="compiled_results.csv",
        help="The compiled csv name that has not yet been filtered.",
    )

    parser.add_argument(
        "--filtered_csv_name",
        type=str,
        default="compiled_results_filtered.csv",
        help="The name of the compiled and filtered csv.",
    )

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--processed_dir_name",
        default="features",
        type=str,
        help="Processed data directory name. Used to store features. Located in data/processed/cnc",
    )

    parser.add_argument(
        "--interim_dir_name",
        type=str,
        default="interim",
        help="Folder name containing all the interim result csv's that will be compiled into one.",
    )

    parser.add_argument(
        "--save_models",
        type=str,
        default="False",
        help="Save the models, and scaler, to disk.",
    )

    parser.add_argument(
        "--check_feat_importance",
        type=str,
        default="False",
        help="Check the feature importance of the models.",
    )

    args = parser.parse_args()

    main(args)
