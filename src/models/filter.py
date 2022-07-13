import pandas as pd
from pathlib import Path
import argparse
import logging
import os
import matplotlib

# run matplotlib without display
# https://stackoverflow.com/a/4706614/9214620
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models.train import train_single_model
from src.models.utils import milling_add_y_label_anomaly, get_model_metrics_df
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

    return proj_dir, path_data_dir, path_model_dir, path_interim_dir, path_final_dir


def filter_results_df(df, keep_top_n=None):
    dfr = df[
        (df["precision_score_min"] > 0)
        & (df["precision_score_max"] < 1)
        & (df["precision_score_std"] > 0)
        & (df["recall_score_min"] > 0)
        & (df["recall_score_max"] < 1)
        & (df["recall_score_std"] > 0)
        & (df["f1_score_min"] > 0)
        & (df["f1_score_max"] < 1)
        & (df["f1_score_std"] > 0)
        & (df["rocauc_min"] < 1)
        & (df["rocauc_max"] < 1)
        & (df["rocauc_avg"] < 1)
        & (df["rocauc_std"] > 0)
        & (df["prauc_min"] < 1)
        & (df["prauc_max"] < 1)
        & (df["prauc_avg"] < 1)
        & (df["prauc_std"] > 0)
        & (df["accuracy_min"] < 1)
        & (df["accuracy_max"] < 1)
        & (df["accuracy_avg"] < 1)
        & (df["accuracy_std"] > 0)
        & (df["n_thresholds_min"] > 3)
        & (df["n_thresholds_max"] > 3)
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
        ]
    return {k: [df.iloc[row_idx][k]] for k in general_param_keys}


def train_and_plot_model_results(df, df_feat, save_n_figures, path_model_curves):
    for row_idx in range(save_n_figures):

        params_clf = rebuild_params_clf(df, row_idx)
        general_params = rebuild_general_params(df, row_idx)

        meta_label_cols = literal_eval(df.iloc[row_idx]["meta_label_cols"])
        stratification_grouping_col = df.iloc[row_idx]["stratification_grouping_col"]
        y_label_col = df.iloc[row_idx]["y_label_col"]
        feat_selection = True
        feat_col_list = literal_eval(df.iloc[row_idx]["feat_col_list"])
        sampler_seed = int(df.iloc[row_idx]["sampler_seed"])
        id = df.iloc[row_idx]["id"]

        (
            model_metrics_dict,
            params_dict_clf_named,
            params_dict_train_setup,
            feat_col_list,
        ) = train_single_model(
            df_feat,
            sampler_seed,
            meta_label_cols,
            stratification_grouping_col,
            y_label_col,
            feat_selection,
            feat_col_list,
            general_params=general_params,
            params_clf=params_clf,
        )

        plot_pr_roc_curves_kfolds(
            model_metrics_dict["precisions_array"],
            model_metrics_dict["recalls_array"],
            model_metrics_dict["fpr_array"],
            model_metrics_dict["tpr_array"],
            model_metrics_dict["rocauc_array"],
            model_metrics_dict["prauc_array"],
            percent_anomalies_truth=0.073,
            path_save_name=path_model_curves / f"curve_{id}.png",
            save_plot=True,
            dpi=300,
        )


def main(args):

    proj_dir, path_data_dir, path_model_dir, path_interim_dir, path_final_dir = set_directories(args)

    df = pd.read_csv(
        path_final_dir / args.compiled_csv_name,
    )
    df = filter_results_df(df)

    # use this is you want to only select the top models by model type (e.g. top SVM, RF, etc.)
    sort_by = 'prauc_avg'
    df = df.groupby(['classifier']).head(int(args.keep_top_n)).sort_values(by=sort_by, ascending=False)

    df.to_csv(path_final_dir / args.filtered_csv_name, index=False)


    # save a certain number of PR-AUC and ROC-AUC curves
    if args.dataset == "milling" and args.save_n_figures > 0:
        assert (
            df.iloc[0]["dataset"] == "milling"
        ), "dataset in results csv is not the milling dataset"

        folder_processed_data_milling = path_data_dir / "processed/milling"

        # load feature dataframe
        df_feat = pd.read_csv(
            folder_processed_data_milling / "milling_features.csv.gz",
            compression="gzip",
        )

        df_feat = milling_add_y_label_anomaly(df_feat)

        path_model_curves = path_final_dir / "model_curves"
        Path(path_model_curves).mkdir(parents=True, exist_ok=True)

        train_and_plot_model_results(df, df_feat, args.save_n_figures, path_model_curves)



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

    args = parser.parse_args()

    main(args)
