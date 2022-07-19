import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from pathlib import Path
import random
import argparse
import logging
import shutil
import random
from datetime import datetime
from ast import literal_eval
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from src.models.utils import (
    milling_add_y_label_anomaly,
    cnc_add_y_label_binary,
    under_over_sampler,
    scale_data,
    calculate_scores,
    collate_scores_binary_classification,
    get_classifier_and_params,
    get_model_metrics_df,
    feat_selection_binary_classification,
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

# turn off debuging for numba
# from https://stackoverflow.com/a/69019168
# logging.getLogger('numba').setLevel(logging.WARNING)


def kfold_cv(
    df,
    clf,
    oversamp_method,
    undersamp_method,
    scaler_method,
    oversamp_ratio,
    undersamp_ratio,
    meta_label_cols,
    stratification_grouping_col=None,
    y_label_col="y",
    n_splits=5,
    feat_selection=None,
    max_feats=None,
    feat_col_list=None,
    early_stopping_rounds=None,
):
    print("feat_select_method: ", feat_selection)
    scores_list = []

    # perform stratified k-fold cross validation using the grouping of the y-label and another column
    if (
        stratification_grouping_col is not None
        and stratification_grouping_col is not y_label_col
    ):
        df_strat = df[[stratification_grouping_col, y_label_col]].drop_duplicates()

        skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # use clone to do a deep copy of model without copying attached data
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html

        for i, (train_index, test_index) in enumerate(
            skfolds.split(
                df_strat[[stratification_grouping_col]], df_strat[[y_label_col]]
            )
        ):
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

            # save x_train_cols to csv file, for debugging
            # Path("x_train_cols.csv").write_text(
            #     "\n".join(x_train_cols)
            # )

            x_train = df_train.values

            # test
            df_test = df[df[stratification_grouping_col].isin(test_strat_vals)]
            y_test = df_test[y_label_col].values.astype(int)
            df_test = df_test.drop(meta_label_cols + [y_label_col], axis=1)
            x_test_cols = df_test.columns
            x_test = df_test.values

            # over-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=oversamp_method, ratio=oversamp_ratio
            )

            # under-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=undersamp_method, ratio=undersamp_ratio
            )

            # scale the data
            x_train, x_test = scale_data(x_train, x_test, scaler_method)

            if feat_selection is not None and i == 0 and feat_col_list is None:
                if feat_selection == "tsfresh":
                    print("performing tsfresh feature selection")
                    x_train, x_test, feat_col_list = feat_selection_binary_classification(
                        x_train,
                        y_train,
                        x_train_cols,
                        x_test,
                        y_test,
                        x_test_cols,
                        feat_col_list=None,
                    )
                elif feat_selection == "tsfresh_random":
                    print("performing tsfresh and random feature selection")
                    _, _, feat_col_list = feat_selection_binary_classification(
                        x_train,
                        y_train,
                        x_train_cols,
                        x_test,
                        y_test,
                        x_test_cols,
                        feat_col_list=None,
                    )

                    num_feats = np.random.randint(low=5, high=len(feat_col_list), size=1)[0]
                    if max_feats is not None and num_feats > max_feats:
                        num_feats = max_feats

                    random_selected_feat = random.sample(list(feat_col_list), num_feats)
                    x_train, x_test, feat_col_list = feat_selection_binary_classification(
                        x_train,
                        y_train,
                        x_train_cols,
                        x_test,
                        y_test,
                        x_test_cols,
                        feat_col_list=random_selected_feat,
                    )


                elif feat_selection == "random":
                    print("performing random feature selection")

                    num_feats = np.random.randint(low=5, high=len(x_train_cols), size=1)[0]
                    if max_feats is not None and num_feats > max_feats:
                        num_feats = max_feats
    
                    random_selected_feat = random.sample(list(x_train_cols), num_feats)
                    x_train, x_test, feat_col_list = feat_selection_binary_classification(
                        x_train,
                        y_train,
                        x_train_cols,
                        x_test,
                        y_test,
                        x_test_cols,
                        feat_col_list=random_selected_feat,
                    )
                else:
                    print("feature selection method not recognized")

            elif feat_selection is not None and feat_col_list is not None:
                print("using already selected features")
                x_train, x_test, feat_col_list = feat_selection_binary_classification(
                    x_train,
                    y_train,
                    x_train_cols,
                    x_test,
                    y_test,
                    x_test_cols,
                    feat_col_list=feat_col_list,
                )
            else:
                print("not using feature selection")
                feat_col_list = list(x_train_cols)

            # can use this to save the feature column names
            # in the results csv when no feature selection is performed
            # if feat_col_list is None:
            #     feat_col_list = list(x_train_cols)

            # train model
            print("x_train shape:", x_train.shape)

            if early_stopping_rounds is not None:
                clone_clf.fit(
                    x_train,
                    y_train,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=[(x_train, y_train), (x_test, y_test)],
                    verbose=True,
                )
            else:
                clone_clf.fit(x_train, y_train)

            # calculate the scores for each individual model train in the cross validation
            # save as a dictionary: "ind_score_dict"
            ind_score_dict = calculate_scores(clone_clf, x_test, y_test)
            scores_list.append(ind_score_dict)

    # perform stratified k-fold cross if only using the y-label for stratification
    else:
        skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(
            skfolds.split(df, df[[y_label_col]])
        ):
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

            # over-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=oversamp_method, ratio=oversamp_ratio
            )

            # under-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=undersamp_method, ratio=undersamp_ratio
            )

            # scale the data
            x_train, x_test = scale_data(x_train, x_test, scaler_method)

            if feat_selection is not None and i == 0 and feat_col_list is None:
                if feat_selection == "tsfresh":
                    print("performing tsfresh feature selection")
                    x_train, x_test, feat_col_list = feat_selection_binary_classification(
                        x_train,
                        y_train,
                        x_train_cols,
                        x_test,
                        y_test,
                        x_test_cols,
                        feat_col_list=None,
                    )
                elif feat_selection == "tsfresh_random":
                    print("performing tsfresh and random feature selection")
                    _, _, feat_col_list = feat_selection_binary_classification(
                        x_train,
                        y_train,
                        x_train_cols,
                        x_test,
                        y_test,
                        x_test_cols,
                        feat_col_list=None,
                    )

                    num_feats = np.random.randint(low=5, high=len(feat_col_list), size=1)[0]
                    if max_feats is not None and num_feats > max_feats:
                        num_feats = max_feats

                    random_selected_feat = random.sample(list(feat_col_list), num_feats)
                    x_train, x_test, feat_col_list = feat_selection_binary_classification(
                        x_train,
                        y_train,
                        x_train_cols,
                        x_test,
                        y_test,
                        x_test_cols,
                        feat_col_list=random_selected_feat,
                    )


                elif feat_selection == "random":
                    print("performing random feature selection")

                    num_feats = np.random.randint(low=5, high=len(x_train_cols), size=1)[0]
                    if max_feats is not None and num_feats > max_feats:
                        num_feats = max_feats

                    random_selected_feat = random.sample(list(x_train_cols), num_feats)
                    x_train, x_test, feat_col_list = feat_selection_binary_classification(
                        x_train,
                        y_train,
                        x_train_cols,
                        x_test,
                        y_test,
                        x_test_cols,
                        feat_col_list=random_selected_feat,
                    )
                else:
                    print("feature selection method not recognized")

            elif feat_selection is not None and feat_col_list is not None:
                print("using already selected features")
                x_train, x_test, feat_col_list = feat_selection_binary_classification(
                    x_train,
                    y_train,
                    x_train_cols,
                    x_test,
                    y_test,
                    x_test_cols,
                    feat_col_list=feat_col_list,
                )
            else:
                print("not using feature selection")
                pass

            # can use this to save the feature column names
            # in the results csv when no feature selection is performed
            # if feat_col_list is None:
            #     feat_col_list = list(x_train_cols)

            if early_stopping_rounds is not None:
                clone_clf.fit(
                    x_train,
                    y_train,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=[(x_train, y_train), (x_test, y_test)],
                )
            else:
                clone_clf.fit(x_train, y_train)

            # calculate the scores for each individual model train in the cross validation
            # save as a dictionary: "ind_score_dict"
            ind_score_dict = calculate_scores(clone_clf, x_test, y_test)
            scores_list.append(ind_score_dict)

    trained_result_dict = collate_scores_binary_classification(scores_list)
    return trained_result_dict, feat_col_list


# TO-DO: need to add the general_params dictionary to the functions.
def train_single_model(
    df,
    sampler_seed,
    meta_label_cols,
    stratification_grouping_col=None,
    y_label_col="y",
    feat_col_list=None,
    general_params=general_params,
    params_clf=None,
):
    # generate the list of parameters to sample over
    params_dict_train_setup = list(
        ParameterSampler(general_params, n_iter=1, random_state=sampler_seed)
    )[0]

    oversamp_method = params_dict_train_setup["oversamp_method"]
    undersamp_method = params_dict_train_setup["undersamp_method"]
    scaler_method = params_dict_train_setup["scaler_method"]
    oversamp_ratio = params_dict_train_setup["oversamp_ratio"]
    undersamp_ratio = params_dict_train_setup["undersamp_ratio"]
    classifier = params_dict_train_setup["classifier"]
    feat_selection = params_dict_train_setup["feat_select_method"]
    max_feats=params_dict_train_setup["max_feats"]


    # if classifier is "xgb" and the 
    if (
        classifier == "xgb"
        and "early_stopping_rounds" in params_dict_train_setup.keys()
        and isinstance(params_dict_train_setup["early_stopping_rounds"], (int, float))
        and params_dict_train_setup["early_stopping_rounds"] >= 0
    ):
        early_stopping_rounds = int(params_dict_train_setup["early_stopping_rounds"])
        print("early_stopping_rounds:", early_stopping_rounds)
        print("type(early_stopping_rounds)", type(early_stopping_rounds))
    else:
        early_stopping_rounds = None

    # add any qualifiers to the parameters. e.g. if "smote_enn", then do not use undersampling
    if oversamp_method in ["smote_enn", "smote_tomek"]:
        undersamp_method = None
        params_dict_train_setup["undersamp_method"] = None

    if oversamp_method == None:
        oversamp_ratio = None
        params_dict_train_setup["oversamp_ratio"] = None

    if undersamp_method == None:
        undersamp_ratio = None
        params_dict_train_setup["undersamp_ratio"] = None

    print(
        f"classifier: {classifier}, oversamp_method: {oversamp_method}, oversamp_ratio: {oversamp_ratio} undersamp_method: {undersamp_method} undersamp_ratio: {undersamp_ratio}"
    )

    # get classifier and its parameters
    clf_function, params_clf_generated = get_classifier_and_params(classifier)

    if params_clf is None:
        params_clf = params_clf_generated

    # instantiate the model
    clf, param_dict_clf_raw, params_dict_clf_named = clf_function(
        sampler_seed, params_clf
    )
    print("\n", params_dict_clf_named)

    model_metrics_dict, feat_col_list = kfold_cv(
        df,
        clf,
        oversamp_method,
        undersamp_method,
        scaler_method,
        oversamp_ratio,
        undersamp_ratio,
        meta_label_cols,
        stratification_grouping_col,
        y_label_col,
        n_splits=5,
        feat_selection=feat_selection,
        max_feats=max_feats,
        feat_col_list=feat_col_list,
        early_stopping_rounds=early_stopping_rounds,
    )

    # added additional parameters to the training setup dictionary
    params_dict_train_setup["sampler_seed"] = sampler_seed

    return (
        model_metrics_dict,
        params_dict_clf_named,
        params_dict_train_setup,
        feat_col_list,
    )


def random_search_runner(
    df,
    rand_search_iter,
    meta_label_cols,
    stratification_grouping_col,
    proj_dir,
    path_save_dir,
    feat_file_name,
    dataset_name=None,
    dataprep_method=None,
    y_label_col="y",
    save_freq=1,
    debug=True,
    feat_col_list=None,
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
                feat_col_list,
            ) = train_single_model(
                df,
                sample_seed,
                meta_label_cols,
                stratification_grouping_col,
                y_label_col,
                feat_col_list,
                general_params=general_params,
                params_clf=None,
            )

            # train setup params
            df_t = pd.DataFrame.from_dict(params_dict_train_setup, orient="index").T
            df_t["feat_col_list"] = str(feat_col_list)
            df_t["meta_label_cols"] = str(meta_label_cols)
            df_t["stratification_grouping_col"] = stratification_grouping_col
            df_t["y_label_col"] = y_label_col
            df_t["feat_file_name"] = str(feat_file_name)
            df_t["n_feats"] = len(feat_col_list)
            df_t["dataprep_method"] = dataprep_method

            # reset feat_col_list
            # can remove this when using "tsfresh" feature selection in order to reuse
            # the feature selection results
            feat_col_list = None

            if args.date_time:
                now_str = str(args.date_time)
            else:
                now = datetime.now()
                now_str = now.strftime("%Y-%m-%d-%H%M-%S")

            df_t["date_time"] = now_str
            df_t["dataset"] = dataset_name
            classifier_used = params_dict_train_setup["classifier"]
            df_t["id"] = f"{sample_seed}_{classifier_used}_{now_str}_{dataset_name}"

            # classifier params
            df_c = pd.DataFrame.from_dict(params_dict_clf_named, orient="index").T

            # model metric results
            df_m = get_model_metrics_df(model_metrics_dict)

            results_list.append(pd.concat([df_t, df_m, df_c], axis=1))

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

    path_processed_dir = path_data_dir / "processed" / args.dataset / args.processed_dir_name
    path_processed_dir.mkdir(parents=True, exist_ok=True)

    return proj_dir, path_data_dir, path_save_dir, path_processed_dir


def train_milling_models(args):

    # set directories
    proj_dir, path_data_dir, path_save_dir, path_processed_dir  = set_directories(args)

    RAND_SEARCH_ITER = args.rand_search_iter

    # load feature dataframe
    if args.feat_file_name:
        feat_file_name = args.feat_file_name
    else:
        feat_file_name = "milling_features.csv"

    df = pd.read_csv(path_processed_dir / feat_file_name)

    # add y label
    df = milling_add_y_label_anomaly(df)

    Y_LABEL_COL = "y"

    # identify if there is another column you want to
    # stratify on, besides the y label
    # STRATIFICATION_GROUPING_COL = "case"
    STRATIFICATION_GROUPING_COL = "cut_no"
    # STRATIFICATION_GROUPING_COL = None

    # list of the columns that are not features columns
    # (not including the y-label column)
    META_LABEL_COLS = ["cut_id", "cut_no", "case", "tool_class"]

    LOG_FILENAME = path_save_dir / "logging_example.out"
    logging.basicConfig(filename=LOG_FILENAME, level=logging.ERROR)

    random_search_runner(
        df,
        RAND_SEARCH_ITER,
        META_LABEL_COLS,
        STRATIFICATION_GROUPING_COL,
        proj_dir,
        path_save_dir,
        feat_file_name,
        dataset_name="milling",
        y_label_col=Y_LABEL_COL,
        save_freq=1,
        debug=True,
    )


def train_cnc_models(args):

    # set directories
    proj_dir, path_data_dir, path_save_dir, path_processed_dir  = set_directories(args)

    RAND_SEARCH_ITER = args.rand_search_iter

    # load feature dataframe
    if args.feat_file_name:
        feat_file_name = args.feat_file_name
    else:
        feat_file_name = "cnc_features_54.csv"

    df = pd.read_csv(path_processed_dir / feat_file_name,)
    df["unix_date"] = df["id"].apply(lambda x: int(x.split("_")[0]))
    df["tool_no"] = df["id"].apply(lambda x: int(x.split("_")[-2]))
    df["index_no"] = df["id"].apply(lambda x: int(x.split("_")[-1]))

    df_labels = pd.read_csv(path_data_dir / "processed/cnc" / "high_level_labels_MASTER_update2020-08-06_new-jan-may-data_with_case.csv")

    # add y label
    df = cnc_add_y_label_binary(df, df_labels, col_list_case=['case_tool_54'])
    df = df.dropna(axis=0)

    Y_LABEL_COL = "y"

    # identify if there is another column you want to
    # stratify on, besides the y label
    # STRATIFICATION_GROUPING_COL = "case"
    STRATIFICATION_GROUPING_COL = "case_tool_54"
    # STRATIFICATION_GROUPING_COL = None

    # list of the columns that are not features columns
    # (not including the y-label column)
    META_LABEL_COLS = ["id", "unix_date", "tool_no", "index_no", "case_tool_54"]

    LOG_FILENAME = path_save_dir / "logging_example.out"
    logging.basicConfig(filename=LOG_FILENAME, level=logging.ERROR)

    random_search_runner(
        df,
        RAND_SEARCH_ITER,
        META_LABEL_COLS,
        STRATIFICATION_GROUPING_COL,
        proj_dir,
        path_save_dir,
        feat_file_name,
        dataset_name="cnc",
        y_label_col=Y_LABEL_COL,
        save_freq=1,
        debug=True,
    )


def main(args):

    if args.dataset == "milling":
        train_milling_models(args)
    elif args.dataset == "cnc":
        train_cnc_models(args)
    else:
        print("no dataset for training specified")
    # elif args.dataset == "cnc":
    #     train_cnc_models(args)


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
        "--processed_dir_name",
        default="features",
        type=str,
        help="Name of the save directory. Used to store features. Located in data/processed/cnc",
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
        help="Name of the dataset to use for training. Either 'milling' or 'cnc'",
    )

    # parser.add_argument(
    #     "--feat_selection",
    #     type=str,
    #     default="False",
    #     help="Conduct feature selection on first iteration",
    # )

    parser.add_argument(
        "--feat_file_name",
        type=str,
        help="Name of the feature file",
    )

    args = parser.parse_args()

    main(args)
