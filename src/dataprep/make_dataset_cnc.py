# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import argparse
import zipfile
import shutil
import pickle
import os
import pandas as pd
import numpy as np
import re
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map


def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    path_splits_dir = path_data_dir / "raw" / "cnc" / args.split_dir_name
    path_processed_raw_dir = path_data_dir / "raw" / "cnc" / args.save_dir_name
    path_processed_raw_dir.mkdir(parents=True, exist_ok=True)


    return proj_dir, path_data_dir, path_splits_dir, path_processed_raw_dir



def read_pickle(filename, col_selection=["current_sub"], rename_cols={"current_sub": "current"}):
    'converts a filename to a pandas dataframe'
    df = pickle.load(open(filename, 'rb'))
    id = filename.stem
    df["id"] = id
    df["time"] = np.arange(0, len(df)) / 1000.0
    df = df[["id", "time"] + col_selection]

    # rename col_selection to with rename_cols dict
    return df.rename(columns=rename_cols)

def main(args):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    proj_dir, path_data_dir, path_splits_dir, path_processed_raw_dir = set_directories(args)

    # get a list of file names
    pattern_match = re.compile(f"\w_{args.tool_no}_")
    files = os.listdir(path_splits_dir)
    file_list = [
        Path(path_splits_dir) / filename
        for filename in files
        if filename.endswith(".pickle") and re.search(pattern_match, filename) is not None
    ]

    # from tqdm creator: https://stackoverflow.com/a/59905309
    df_list = process_map(read_pickle, file_list, max_workers=args.n_cores, chunksize=1000)

    return df_list


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    parser = argparse.ArgumentParser(description="Create dataframe from raw data")

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--path_data_dir",
        type=str,
        default="data/",
        help="Path to data folder that contains raw/interim/processed data folders",
    )

    parser.add_argument(
        "--save_dir_name",
        type=str,
        default="data_raw_processed",
        help="Name of the subfolder that will contain the final raw csv file",
    )

    parser.add_argument(
        "--split_dir_name",
        type=str,
        default="data_splits",
        help="Folder name containing all individual split pickle files.",
    )

    parser.add_argument(
        "--n_cores",
        type=int,
        default=2,
        help="Number of processes to use for multiprocessing",
    )

    parser.add_argument(
        "--tool_no",
        type=int,
        default=54,
        help="Tool number to save to the dataframe",
    )


    args = parser.parse_args()

    proj_dir, path_data_dir, path_splits_dir, path_processed_raw_dir = set_directories(args)

    df_list = main(args)

    df = pd.concat(df_list, ignore_index=True)

    print("Shape of final df:", df.shape)

    # save the dataframe
    print("Saving dataframe...")
    df.to_csv(
        path_processed_raw_dir / f"cnc_raw_{args.tool_no}.csv",
        index=False,
        )


    shutil.copy(
        Path(proj_dir) / "src/dataprep/make_dataset_cnc.py",
        path_processed_raw_dir / "make_dataset_cnc.py",)
