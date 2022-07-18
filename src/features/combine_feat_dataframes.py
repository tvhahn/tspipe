from pathlib import Path
import pandas as pd
from multiprocessing import Pool
import os
import logging
import argparse
from src.features.utils import set_directories


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    return pd.read_csv(filename)


def main(args):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    proj_dir, path_data_dir, path_raw_dir, path_interim_dir, path_processed_dir = set_directories(args)

    # get a list of file names
    files = os.listdir(path_interim_dir)
    file_list = [
        Path(path_interim_dir) / filename
        for filename in files
        if filename.endswith(".csv")
    ]

    # set up your pool
    with Pool(processes=args.n_cores) as pool:  # or whatever your hardware can support

        # have your pool map the file names to dataframes
        df_list = pool.map(read_csv, file_list)

        # reduce the list of dataframes to a single dataframe
        combined_df = pd.concat(df_list, ignore_index=True)

        return combined_df


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("combine multiple feature dataframes (in CSV format) into one")


    parser = argparse.ArgumentParser(description="Build features")

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
        help="Path to data folder that contains raw/interim/processed data folders",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to use for training. Either 'milling' or 'cnc'",
    )

    parser.add_argument(
        "--feat_file_name",
        type=str,
        help="Name of the final feature file.",
    )

    parser.add_argument(
        "--interim_dir_name",
        default="features",
        type=str,
        help="Name of the save directory. Used to store features when processing in chunks. Located in data/interim/cnc",
    )

    parser.add_argument(
        "--processed_dir_name",
        default="features",
        type=str,
        help="Name of the save directory. Used to store features. Located in data/processed/cnc",
    )

    parser.add_argument(
        "--n_cores", type=int, default=2, help="Number of cores to use"
    )

    args = parser.parse_args()



    df = main(args)
    print("Final df shape:", df.shape)

    proj_dir, path_data_dir, path_raw_dir, path_interim_dir, path_processed_dir = set_directories(args)

    df.to_csv(path_processed_dir / args.feat_file_name, index=False)