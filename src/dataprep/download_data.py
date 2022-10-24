import logging
from pathlib import Path
import argparse
from src.dataprep.utils import set_directories
from pyphm.datasets.milling import MillingPrepMethodA


def main(args):
    """Download the datasets."""

    logger = logging.getLogger(__name__)
    logger.info("Download the datasets")

    proj_dir, path_data_dir, path_raw_dir = set_directories(args)

    print("Downloading milling data...")
    milldata = MillingPrepMethodA(
        root = path_raw_dir, download=True)
    


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
        help="Path to data folder that contains raw/interim/processed data folders",
    )

    args = parser.parse_args()

    main(args)
