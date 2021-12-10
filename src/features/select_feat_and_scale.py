from pathlib import Path
import pandas as pd
from multiprocessing import Pool
import os
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tsfresh import select_features, feature_extraction
import pickle
import json

"""
Script to select the "optimum" features from the features dataframe.

    - The features dataframe is a pandas dataframe that contains all the features

Note: it is important that the appropriate data splits are performed before selecting 
features. Otherwise, you will be injecting label information into the val/test sets
(aka. data leakage).
"""

###############################################################################
# Set arguments
###############################################################################

# parser = argparse.ArgumentParser(description="Build features")

# parser.add_argument(
#     "--path_data_folder",
#     type=str,
#     default="data/",
#     help="Path to data folder that contains raw/interim/processed data folders",
# )

# parser.add_argument(
#     "--num_pool_processes",
#     type=int,
#     default=2,
#     help="Number of processes to use for multiprocessing",
# )

# args = parser.parse_args()

# path_data_folder = Path(args.path_data_folder)
# num_pool_processes = int(args.num_pool_processes)


###############################################################################
# Functions
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


def milling_stratify_cut_no(df, random_seed=76, train_size_pct=0.4):
    """
    Put the unique cut_no's into lists for the train/test/val splits. Ensure
    stratification is maintained using the tool_class column of the dataframe.
    """

    # dictionary comprehension to create dict of each cut_no and its corresponding tool_class
    cut_id_tool_class_dict = {
        i: int(df[df["cut_no"] == i]["tool_class"].unique())
        for i in df["cut_no"].unique()
    }

    cut_numbers = []
    tool_classes = []
    for i in cut_id_tool_class_dict:
        cut_numbers.append(i)
        tool_classes.append(cut_id_tool_class_dict[i])

    (
        cut_numbers_train,
        cut_numbers_test,
        tool_classes_train,
        tool_classes_test,
    ) = train_test_split(
        np.array(cut_numbers),
        np.array(tool_classes),
        test_size=1 - train_size_pct,
        random_state=random_seed,
        stratify=np.array(tool_classes),
    )

    (
        cut_numbers_val,
        cut_numbers_test,
        tool_classes_val,
        tool_classes_test,
    ) = train_test_split(
        cut_numbers_test,
        tool_classes_test,
        test_size=0.5,
        random_state=random_seed,
        stratify=tool_classes_test,
    )

    # assert that there are no duplicates between the cut_numbers_train/val/test
    assert (
        len(np.intersect1d(cut_numbers_train, cut_numbers_test)) == 0
    ), "Duplicate cut_no's found in train/test split"
    assert (
        len(np.intersect1d(cut_numbers_train, cut_numbers_val)) == 0
    ), "Duplicate cut_no's found in train/val split"
    assert (
        len(np.intersect1d(cut_numbers_val, cut_numbers_test)) == 0
    ), "Duplicate cut_no's found in val/test split"

    return (
        cut_numbers_train,
        cut_numbers_val,
        cut_numbers_test,
    )


def milling_train_test_split(
    df_feat, cut_numbers_train, cut_numbers_val, cut_numbers_test
):
    """
    Create the train/val/test splits for the features dataframe.
    """

    df_train = df_feat[df_feat["cut_no"].isin(cut_numbers_train)]
    df_val = df_feat[df_feat["cut_no"].isin(cut_numbers_val)]
    df_test = df_feat[df_feat["cut_no"].isin(cut_numbers_test)]

    return (df_train, df_val, df_test)


def milling_select_features(df_feat, random_seed=76, train_size_pct=0.4):
    """
    Select the optimum features from the features dataframe.

    Parameters
    ----------
    df_feat : pandas dataframe
        Features dataframe. Must contain the "cut_id", "cut_no", "case",
        and "tool_class" columns (along with feature columns)

    random_seed : int
        Random seed for the train/test/val splits

    train_size_pct : float
        Percentage of the dataframe to use for the train split

    Returns
    -------
    df_train, df_val, df_test : pandas dataframes
        Train/val/test splits of the features dataframe, with the features
        selected by the tsfresh library.

    """

    df_feat = df_feat.reset_index(drop=True)  # reset index just in case

    print("Original shape of df_feat:", df_feat.shape)

    # generate cut_numbers_train/val/test from entire dataframe
    cut_numbers_train, cut_numbers_val, cut_numbers_test = milling_stratify_cut_no(
        df_feat, random_seed, train_size_pct
    )

    # add y label to the features dataframe
    df_feat = milling_add_y_label_anomaly(df_feat)

    # remove NaN values
    df_feat = df_feat.dropna(axis=1, how="any")

    # if index not set to cut_id, then set it
    # this is the needed format for the tsfresh library
    if pd.Index(np.arange(0, df_feat.shape[0])).equals(df_feat.index):
        df_feat = df_feat.set_index("cut_id")

    df_train, df_val, df_test = milling_train_test_split(
        df_feat, cut_numbers_train, cut_numbers_val, cut_numbers_test
    )

    # assert that df_feat has a y column
    assert "y" in df_feat.columns, "df_feat must have a y column"

    # perform the feature selection on df_train
    df_train_feat_sel = select_features(
        df_train.drop(["cut_no", "case", "tool_class", "y"], axis=1),
        df_train["y"],
        n_jobs=5,
        chunksize=10,
    )

    col_selected = list(df_train_feat_sel.columns)
    col_selected = col_selected + ["cut_no", "case", "tool_class", "y"]

    (df_train, df_val, df_test) = (
        df_train[col_selected],
        df_val[col_selected],
        df_test[col_selected],
    )

    # assert that df_train/val/test have the same columns
    assert df_train.columns.equals(df_val.columns) and df_train.columns.equals(
        df_test.columns
    ), "Columns must be the same for all dataframes"

    # print shapes of the train/val/test splits and the percentage or rows compared to the combined total
    print(
        "Final df_train shape:",
        df_train.shape,
        f"({df_train.shape[0] / df_feat.shape[0] * 100:.2f}% of samples)",
    )
    print(
        "Final df_val shape:",
        df_val.shape,
        f"({df_val.shape[0] / df_feat.shape[0] * 100:.2f}% of samples)",
    )
    print(
        "Final df_test shape:",
        df_test.shape,
        f"({df_test.shape[0] / df_feat.shape[0] * 100:.2f}% of samples)",
    )

    # return the df_train/val/test splits and reset the index for each dataframe
    return (
        df_train.reset_index(),
        df_val.reset_index(),
        df_test.reset_index(),
    )


def save_feat_dict(df_feat, col_ignore_list, folder_save_path, save_name="feat_dict"):
    """
    Save the features dataframe as a dictionary.

    Parameters
    ----------
    df_feat : pandas dataframe
        Features dataframe.

    folder_save_path : pathlib.Path
        Folder path where the dictionary will be saved as a json file.

    save_name : str
        prefix of the file name.

    """

    feat_dict = feature_extraction.settings.from_columns(
        df_feat.columns, col_ignore_list
    )  # create the feature dictionary

    # save the dictionary to a json file
    with open(folder_save_path / f"{save_name}.json", "w") as f:
        json.dump(feat_dict, f)

    # save the col_selected list to a .txt file
    with open(folder_save_path / f"{save_name}_col_list.txt", "w") as f:
        f.write("\n".join(list(df_feat.drop(col_ignore_list, axis=1).columns)))


# function to scale the df_train, df_val, and df_test dataframes with the same scaler
def scale_dataframes(
    df_train,
    df_val,
    df_test,
    scaler,
    col_drop_list,
    col_y_labels,
    save_data=True,
    folder_save_path=None,
):
    """
    Scale the train/val/test dataframes with the same scaler.

    Parameters
    ----------
    df_train : pandas dataframe
        Train dataframe with all the features, y labels, etc.

    df_val : pandas dataframe
        Validation dataframe.

    df_test : pandas dataframe
        Test dataframe.

    scaler : sklearn scaler object
        Scaler object to scale the dataframes.

    Returns
    -------
    df_train, df_val, df_test : pandas dataframes
        Scaled train/val/test dataframes.



    """

    # assert that df_train/val/test do not have col_y_labels in the columns
    # assert not any(
    #     col_y_labels in df_train.columns
    # ), "df_train must not have col_y_labels in the columns"

    x_train = scaler.transform(df_val.drop(col_drop_list, axis=1))
    x_val = scaler.transform(df_val.drop(col_drop_list, axis=1))
    x_test = scaler.transform(df_test.drop(col_drop_list, axis=1))

    # save the x_train, x_val, and x_test arrays as .npy files
    if save_data:
        np.save(folder_save_path / "x_train.npy", x_train)
        np.save(folder_save_path / "x_val.npy", x_val)
        np.save(folder_save_path / "x_test.npy", x_test)

    # save the y_train, y_val, and y_test arrays as .npy files
    # and include the "cut_no", "case", "y", and "tool_class" columns
    y_train = df_train.reset_index()[col_y_labels].to_numpy()
    y_val = df_val.reset_index()[col_y_labels].to_numpy()
    y_test = df_test.reset_index()[col_y_labels].to_numpy()

    if save_data:
        np.save(folder_save_path / "y_train.npy", y_train)
        np.save(folder_save_path / "y_val.npy", y_val)
        np.save(folder_save_path / "y_test.npy", y_test)

    return df_train, df_val, df_test, y_train, y_val, y_test


def main(path_data_folder):
    """
    Runs the feature selection and scaling on the data.
    """
    logger = logging.getLogger(__name__)
    logger.info("making train/val/test data sets from features")

    ######
    # UC Berkeley milling data
    ######
    folder_raw_data_milling = path_data_folder / "raw/milling"
    folder_interim_data_milling = path_data_folder / "interim/milling"
    folder_processed_data_milling = path_data_folder / "processed/milling"

    # read in the milling features to a pandas dataframe
    df_feat = pd.read_csv(
        folder_processed_data_milling / "milling_features.csv.gz",
        compression="gzip",
    )

    # select the features and put into df_train, df_val, and df_test
    df_train, df_val, df_test = milling_select_features(
        df_feat, random_seed=76, train_size_pct=0.40
    )

    # save the feature dictionary to json and the col_selected list to txt
    # columns not to include in x_train/x_val/x_test
    col_drop_list = [
        "cut_id",
        "cut_no",
        "case",
        "tool_class",
        "y",
    ]  
    # columns to be used as y labels
    col_y_labels = [
        "cut_id",
        "cut_no",
        "case",
        "tool_class",
        "y",
    ]

    save_feat_dict(
        df_train, col_drop_list, folder_processed_data_milling, save_name="feat_dict"
    )

    # scale the dataframes
    scaler = preprocessing.StandardScaler().fit(
        df_train.drop(col_drop_list, axis=1)
    )

    (x_train, x_val, x_test, y_train, y_val, y_test) = scale_dataframes(
        df_train,
        df_val,
        df_test,
        scaler,
        col_drop_list,
        col_y_labels,
        save_data=True,
        folder_save_path=folder_processed_data_milling,
    )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Select the optimal features, scale the data, and save the train/val/test splits.")

    parser = argparse.ArgumentParser(description="Select features, scale the data, and save.")

    parser.add_argument(
        "--path_data_folder",
        type=str,
        default="data/",
        help="Path to data folder that contains raw/interim/processed data folders",
    )

    args = parser.parse_args()

    path_data_folder = Path(args.path_data_folder)

    main(path_data_folder)