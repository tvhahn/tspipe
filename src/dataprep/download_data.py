# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import argparse
import zipfile
# from src.datasets.milling import MillingDataPrep
from pyphm.datasets.milling import MillingPrepMethodA


def main(path_data_folder):
    """Download the datasets."""

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    path_data_raw_folder = path_data_folder / "raw"

    print("Downloading milling data...")
    milldata = MillingPrepMethodA(
        root = path_data_raw_folder, download=True)
    


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)


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

    main(path_data_folder)
