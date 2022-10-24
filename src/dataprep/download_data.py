import logging
from pathlib import Path
import argparse
from src.dataprep.utils import set_directories
from pyphm.datasets.milling import MillingPrepMethodA
import zipfile


def main(args):
    """Download the datasets."""

    logger = logging.getLogger(__name__)
    logger.info("Download the datasets")

    proj_dir, path_data_dir, path_raw_dir = set_directories(args)

    
    path_mill_zip = path_raw_dir / "milling" / "mill.zip"
    path_mill_mat = path_raw_dir / "milling" / "mill.mat"

    # if mill.zip exists and mill.mat does not exist, then unzip mill.zip
    if path_mill_zip.exists() and not path_mill_mat.exists():
        with zipfile.ZipFile(path_mill_zip, "r") as zip_ref:
            zip_ref.extractall(path_raw_dir / "milling")

    # if mill.zip exists and mill.mat exists, then pass
    elif path_mill_zip.exists() and path_mill_mat.exists():
        pass
    
    # else, download mill.zip
    else:
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
