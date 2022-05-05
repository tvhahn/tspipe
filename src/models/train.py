import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from pathlib import Path
import random
import argparse
import logging
import shutil
from datetime import datetime
from sklearn.base import clone
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
    feat_selection=False,
    feat_col_list=None,
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
        

        for i, (train_index, test_index) in enumerate(skfolds.split(
            df_strat[[stratification_grouping_col]], df_strat[[y_label_col]]
        )):
            print(i)

            clone_clf = clone(clf)
            train_strat_vals = df_strat.iloc[train_index][
                stratification_grouping_col
            ].values

            test_strat_vals = df_strat.iloc[test_index][
                stratification_grouping_col
            ].values

            # train
            df_train = df[df[stratification_grouping_col].isin(train_strat_vals)]
            y_train = df_train[y_label_col].values.astype(int)
            df_train = df_train.drop(meta_label_cols + [y_label_col], axis=1)
            x_train_cols = df_train.columns
            x_train = df_train.values

            # test
            df_test = df[df[stratification_grouping_col].isin(train_strat_vals)]
            y_test = df_test[y_label_col].values.astype(int)            
            df_test = df_test.drop(meta_label_cols + [y_label_col], axis=1)
            x_test_cols = df_test.columns
            x_test = df_test.values
            

            # scale the data
            x_train, x_test = scale_data(x_train, x_test, scaler_method)

            # do feature selection if specified
            if feat_selection and i==0 and feat_col_list is None:
                from tsfresh import select_features # import in loop because it is a heavy package
                print("Performing feature selection")

                x_train = select_features(
                    pd.DataFrame(x_train, columns=x_train_cols),
                    y_train,
                    n_jobs=5,
                    chunksize=10, ml_task="classification", multiclass=False)
                
                feat_col_list = list(x_train.columns)

                x_train = x_train.values
                x_test = pd.DataFrame(x_test, columns=x_test_cols)[feat_col_list].values
            elif feat_selection and feat_col_list is not None:
                x_train = pd.DataFrame(x_train, columns=x_train_cols)[feat_col_list].values
                x_test = pd.DataFrame(x_test, columns=x_test_cols)[feat_col_list].values
            else:
                pass  # no feature selection


            # under-over-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=uo_method, ratio=imbalance_ratio
            )

            # train model
            print("x_train shape:", x_train.shape)
            clone_clf.fit(x_train, y_train)

            # calculate the scores for each individual model train in the cross validation
            # save as a dictionary: "ind_score_dict"
            ind_score_dict = calculate_scores(
                clone_clf, 
                x_test, 
                y_test)

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

        for i, (train_index, test_index) in enumerate(skfolds.split(df, df[[y_label_col]])):
            clone_clf = clone(clf)

            df_train = df.iloc[train_index]
            df_test = df.iloc[test_index]

            y_train = df_train[y_label_col].values.astype(int)
            df_train = df_train.drop(meta_label_cols + [y_label_col], axis=1)
            x_train_cols = df_train.columns
            x_train = df_train.values

            y_test = df_test[y_label_col].values.astype(int)
            df_test = df_test.drop(meta_label_cols + [y_label_col], axis=1)
            x_test_cols = df_test.columns
            x_test = df_test.values

            # scale the data
            scale_data(x_train, x_test, scaler_method)

                        # do feature selection if specified
            if feat_selection and i==0 and feat_col_list is None:
                from tsfresh import select_features # import in loop because it is a heavy package
                print("Performing feature selection")

                x_train = select_features(
                    pd.DataFrame(x_train, columns=x_train_cols),
                    y_train,
                    n_jobs=5,
                    chunksize=10, ml_task="classification", multiclass=False)
                
                feat_col_list = list(x_train.columns)

                x_train = x_train.values
                x_test = pd.DataFrame(x_test, columns=x_test_cols)[feat_col_list].values
            elif feat_selection and feat_col_list is not None:
                x_train = pd.DataFrame(x_train, columns=x_train_cols)[feat_col_list].values
                x_test = pd.DataFrame(x_test, columns=x_test_cols)[feat_col_list].values
            else:
                pass  # no feature selection

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

    return trained_result_dict, feat_col_list

# TO-DO: need to add the general_params dictionary to the functions.
def train_single_model(
    df, sampler_seed, meta_label_cols, stratification_grouping_col=None, y_label_col="y", feat_selection=False, feat_col_list=None
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

    model_metrics_dict, feat_col_list = kfold_cv(
        df,
        clf,
        uo_method,
        scaler_method,
        imbalance_ratio,
        meta_label_cols,
        stratification_grouping_col,
        y_label_col,
        n_splits=5,
        feat_selection=feat_selection,
        feat_col_list=feat_col_list
    )

    # added additional parameters to the training setup dictionary
    params_dict_train_setup["sampler_seed"] = sampler_seed

    return model_metrics_dict, params_dict_clf_named, params_dict_train_setup, feat_col_list


def random_search_runner(
    df,
    rand_search_iter,
    meta_label_cols,
    stratification_grouping_col,
    proj_dir,
    path_save_dir,
    feat_selection,
    dataset_name=None,
    y_label_col="y",
    save_freq=1,
    debug=False,
    feat_col_list=None
):

    results_list = []
    for i in range(rand_search_iter):
        # set random sample seed
        sample_seed = random.randint(0, 2 ** 25)
        # sample_seed = 13

        if i == 0:
            file_name_results = f"results_{sample_seed}.csv"

            # copy the random_search_setup.py file to path_save_dir using shutil if it doesn't exist
            if (path_save_dir / "setup_files" / "random_search_setup.py").exists():
                pass
            else:
                shutil.copy(
                    proj_dir / "src/models/random_search_setup.py",
                    path_save_dir / "setup_files" / "random_search_setup.py",
                )

        try:

            (
                model_metrics_dict,
                params_dict_clf_named,
                params_dict_train_setup,
                feat_col_list
            ) = train_single_model(
                df,
                sample_seed,
                meta_label_cols,
                stratification_grouping_col,
                y_label_col,
                feat_selection,
                feat_col_list
            )

            # train setup params
            df_t = pd.DataFrame.from_dict(params_dict_train_setup, orient="index").T
            df_t["feat_col_list"] = str(feat_col_list)

            if args.date_time:
                now_str = str(args.date_time)
            else:
                now = datetime.now()
                now_str = now.strftime("%Y-%m-%d-%H%M-%S")

            df_t['date_time'] = now_str
            df_t['dataset'] = dataset_name

            # classifier params
            df_c = pd.DataFrame.from_dict(params_dict_clf_named, orient="index").T

            # model metric results
            df_m = get_model_metrics_df(model_metrics_dict)

            results_list.append(
                pd.concat([df_t, df_m, df_c], axis=1)
            )  

            if i % save_freq == 0:
                df_results = pd.concat(results_list)

                if path_save_dir is not None:
                    df_results.to_csv(path_save_dir / file_name_results, index=False)
                else:
                    df_results.to_csv(file_name_results, index=False)

        # except Exception as e and log the exception
        except Exception as e:
            if debug:
                print("####### Exception #######")
                print(e)
                logging.exception(f"##### Exception in random_search_runner:\n{e}\n\n")
            pass

def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    save_dir_name = args.save_dir_name


    # check if "scratch" path exists in the home directory
    # if it does, assume we are on HPC
    
    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        print("Assume on HPC")

        path_save_dir = scratch_path / "feat-store/models" / args.save_dir_name
        Path(path_save_dir / "setup_files").mkdir(parents=True, exist_ok=True)

    else:
        print("Assume on local compute")
        path_save_dir = proj_dir / "models" / save_dir_name
        Path(path_save_dir / "setup_files").mkdir(parents=True, exist_ok=True)

    return proj_dir, path_data_dir, path_save_dir


def main(args):

    # set directories
    proj_dir, path_data_dir, path_save_dir = set_directories(args)

    folder_raw_data_milling = path_data_dir / "raw/milling"
    folder_interim_data_milling = path_data_dir / "interim/milling"
    folder_processed_data_milling = path_data_dir / "processed/milling"
    folder_models = proj_dir / "models"

    RAND_SEARCH_ITER = args.rand_search_iter

    # set a seed for the parameter sampler
    # SAMPLER_SEED = random.randint(0, 2 ** 16)

    # load feature dataframe
    df = pd.read_csv(
        folder_processed_data_milling / "milling_features.csv.gz", compression="gzip"
    )

    # add y label
    df = milling_add_y_label_anomaly(df)

    Y_LABEL_COL = "y"

    # identify if there is another column you want to
    # stratify on, besides the y label
    STRATIFICATION_GROUPING_COL = "cut_no"
    # STRATIFICATION_GROUPING_COL = None

    # list of the columns that are not features columns
    # (not including the y-label column)
    META_LABEL_COLS = ["cut_id", "cut_no", "case", "tool_class"]

    LOG_FILENAME = path_save_dir / "logging_example.out"
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    random_search_runner(
        df,
        RAND_SEARCH_ITER,
        META_LABEL_COLS,
        STRATIFICATION_GROUPING_COL,
        proj_dir,
        path_save_dir,
        feat_selection=args.feat_selection,
        dataset_name="milling",
        y_label_col="y",
        save_freq=1,
        debug=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build data sets for analysis")

    parser.add_argument(
        "--n_cores",
        type=int,
        default=1,
        help="Number of cores to use for multiprocessing",
    )

    parser.add_argument(
        "--rand_search_iter",
        type=int,
        default=2,
        help="Number number of randem search iterations",
    )


    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--path_data_dir",
        dest="path_data_dir",
        type=str,
        help="Location of the data folder, containing the raw, interim, and processed folders",
    )

    parser.add_argument(
        "--save_dir_name",
        default="interim_results",
        type=str,
        help="Name of the save directory. Used to store the results of the random search",
    )

    parser.add_argument(
        "--date_time",
        type=str,
        help="Date and time that random search was executed.",
    )

    parser.add_argument(
        "--dataset",
        default="milling",
        type=str,
        help="Date and time that random search was executed.",
    )

    parser.add_argument(
        "--feat_selection",
        default=False,
        action="store_true",
        help="Conduct feature selection on first iteration",
    )



    args = parser.parse_args()

    main(args)
