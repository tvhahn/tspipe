# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import argparse
import zipfile
from src.data.data_prep_utils import MillingDataPrep

parser = argparse.ArgumentParser(description="Create dataframe from raw data")

parser.add_argument(
    "--path_data_folder",
    type=str,
    default="data/",
    help="Path to data folder that contains raw/interim/processed data folders",
)

args = parser.parse_args()

path_data_folder = Path(args.path_data_folder)
print("path_data_folder:", path_data_folder)


def main(path_data_folder):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    folder_raw_data_milling = path_data_folder / "raw/milling"
    folder_processed_data_milling = path_data_folder / "processed/milling"
    df_label_path = folder_processed_data_milling / "labels_with_tool_class.csv"

    # extract mill.zip file in folder_raw_data_milling if it hasn't been extracted yet
    if not (folder_raw_data_milling / "mill.mat").exists():
        print("Extracting mill.zip file...")
        with zipfile.ZipFile(folder_raw_data_milling / 'mill.zip', 'r') as zip_ref:
            zip_ref.extractall(folder_raw_data_milling)

    print("Creating dataframe from raw data...")
    milldata = MillingDataPrep(
        folder_raw_data_milling / "mill.mat",
        path_df_labels=df_label_path,
        window_size=64,
        stride=64,
    )
    
    df = milldata.create_xy_dataframe()
    print("Shape of final df:", df.shape)

    # save the dataframe
    print("Saving dataframe...")
    df.to_csv(
        folder_raw_data_milling / "milling.csv.gz",
        compression="gzip",
        index=False,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(path_data_folder)
