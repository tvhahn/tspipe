from pathlib import Path
import pandas as pd
from multiprocessing import Pool
import os
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tsfresh import select_features

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
        test_size= 1 - train_size_pct,
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

    # print the shapes of the train/test/val splits
    print("cut_numbers_train shape:", len(cut_numbers_train))
    print("cut_numbers_test shape:", len(cut_numbers_test))
    print("cut_numbers_val shape:", len(cut_numbers_val))

    # assert that there are no duplicates between the cut_numbers_train/val/test
    assert len(np.intersect1d(cut_numbers_train, cut_numbers_test)) == 0, "Duplicate cut_no's found in train/test split"
    assert len(np.intersect1d(cut_numbers_train, cut_numbers_val)) == 0, "Duplicate cut_no's found in train/val split"
    assert len(np.intersect1d(cut_numbers_val, cut_numbers_test)) == 0, "Duplicate cut_no's found in val/test split"

    return (
        cut_numbers_train,
        cut_numbers_val,
        cut_numbers_test,
    )


def milling_train_test_split(df_feat, cut_numbers_train, cut_numbers_val, cut_numbers_test):
    """
    Create the train/val/test splits for the features dataframe.
    """

    df_train = df_feat[df_feat["cut_no"].isin(cut_numbers_train)]
    df_val = df_feat[df_feat["cut_no"].isin(cut_numbers_val)]
    df_test = df_feat[df_feat["cut_no"].isin(cut_numbers_test)]

    return (df_train, df_val, df_test)



def milling_select_features(df_feat, random_seed=76, train_size_pct=0.4):
    """
    Select the optimum features from the features df
    """

    df_feat = df_feat.reset_index(drop=True)  # reset index just in case

    # generate cut_numbers_train/val/test from entire dataframe
    cut_numbers_train, cut_numbers_val, cut_numbers_test = milling_stratify_cut_no(df_feat, random_seed, train_size_pct)


    # add y label to the features dataframe
    df_feat = milling_add_y_label_anomaly(df_feat)

    # remove NaN values
    df_feat = df_feat.dropna(axis=1, how="any")

    # if index not set to cut_id, then set it
    if pd.Index(np.arange(0, df_feat.shape[0])).equals(df_feat.index):
        df_feat = df_feat.set_index("cut_id")

    df_train, df_val, df_test = milling_train_test_split(df_feat, cut_numbers_train, cut_numbers_val, cut_numbers_test)

    # print the shapes of the train/val/test splits
    print("df_train shape:", df_train.shape)
    print("df_val shape:", df_val.shape)
    print("df_test shape:", df_test.shape)

    # assert that df_feat has a y column
    assert "y" in df_feat.columns, "df_feat must have a y column"

    # perform the feature selection on df_train
    df_train_feat_sel = select_features(df_train.drop(['case', 'tool_class', 'y'], axis=1), df_train["y"], n_jobs=5, chunksize=10)

    col_selected = list(df_train_feat_sel.columns)
    col_selected = col_selected + ['case', 'tool_class', 'y']

    return df_val[col_selected]



def main(folder_interim_data):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # get a list of file names
    files = os.listdir(folder_interim_data)
    file_list = [
        Path(folder_interim_data) / filename
        for filename in files
        if filename.endswith(".csv")
    ]

    # set up your pool
    with Pool(
        processes=num_pool_processes
    ) as pool:  # or whatever your hardware can support

        # have your pool map the file names to dataframes
        df_list = pool.map(read_csv, file_list)

        # reduce the list of dataframes to a single dataframe
        combined_df = pd.concat(df_list, ignore_index=True)

        return combined_df


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("combine multiple feature dataframes (in CSV format) into one")

    ### Milling data ###
    folder_raw_data_milling = path_data_folder / "raw/milling"
    folder_interim_data_milling = path_data_folder / "interim/milling"
    folder_processed_data_milling = path_data_folder / "processed/milling"

    df = main(folder_interim_data_milling)
    print("Final df shape:", df.shape)

    df.to_csv(
        folder_processed_data_milling / "milling_features.csv.gz",
        compression="gzip",
        index=False,
    )
