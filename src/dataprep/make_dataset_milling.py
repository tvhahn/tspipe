# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import argparse
import zipfile
import shutil
from pyphm.datasets.milling import MillingPrepMethodA


def main(args):
    """Main function to prepare the milling dataset. Segments the signals
    into windows and saves them as a csv file.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating windows of the milling dataset")

    path_data_dir = Path(args.path_data_dir)
    print("path_data_dir:", path_data_dir)

    folder_raw_data = path_data_dir / "raw"
    folder_raw_data_milling = path_data_dir / "raw/milling"

    sub_folder_path = folder_raw_data_milling / args.raw_dir_name
    Path(sub_folder_path).mkdir(parents=True, exist_ok=True)

    folder_processed_data_milling = path_data_dir / "processed/milling"
    path_csv_labels = (
        folder_processed_data_milling / "milling_labels_with_tool_class.csv"
    )

    # extract mill.zip file in folder_raw_data_milling if it hasn't been extracted yet
    if not (folder_raw_data_milling / "mill.mat").exists():
        print("Extracting mill.zip file...")
        with zipfile.ZipFile(folder_raw_data_milling / "mill.zip", "r") as zip_ref:
            zip_ref.extractall(folder_raw_data_milling)

    milldata = MillingPrepMethodA(
        root=folder_raw_data,
        window_len=args.window_len,
        stride=args.stride,
        download=True,
    )

    df = milldata.create_xy_dataframe()
    print("Shape of final df:", df.shape)

    # save the dataframe
    print("Saving dataframe as milling.csv.gz ...")
    df.to_csv(
        sub_folder_path / "milling.csv.gz",
        compression="gzip",
        index=False,
    )

    shutil.copy(
        Path(args.proj_dir) / "src/dataprep/make_dataset_milling.py",
        sub_folder_path / "make_dataset_milling.py",
    )


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
        "--raw_dir_name",
        type=str,
        default="data_raw_processed",
        help="Name of the subfolder that will contain the final raw csv file",
    )

    parser.add_argument(
        "--window_len",
        type=int,
        default=64,
        help="Window length for the sample.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Stride between each sample.",
    )

    args = parser.parse_args()

    main(args)
