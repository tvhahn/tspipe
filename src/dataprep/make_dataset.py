# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import argparse
import zipfile
import shutil
from pyphm.datasets.milling import MillingPrepMethodA

# def set_directories(args):

#     if args.proj_dir:
#         proj_dir = Path(args.proj_dir)
#     else:
#         proj_dir = Path().cwd()

#     if args.path_data_dir:
#         path_data_dir = Path(args.path_data_dir)
#     else:
#         path_data_dir = proj_dir / "data"

#     sub_folder_name = args.sub_folder_name
    
#     scratch_path = Path.home() / "scratch"
#     if scratch_path.exists():
#         print("Assume on HPC")

#         sub_folder_dir = scratch_path / "feat-store/data/raw" / sub_folder_name

#     else:
#         print("Assume on local compute")
#         sub_folder_dir = proj_dir / "models" / final_dir_name

#     return proj_dir, path_data_dir, path_final_dir


def main(args):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    path_data_dir = Path(args.path_data_dir)
    print("path_data_dir:", path_data_dir)

    folder_raw_data = path_data_dir / "raw"
    folder_raw_data_milling = path_data_dir / "raw/milling" 

    sub_folder_path = folder_raw_data_milling / args.sub_folder_name
    Path(sub_folder_path).mkdir(parents=True, exist_ok=True)

    folder_processed_data_milling = path_data_dir / "processed/milling"
    path_csv_labels = folder_processed_data_milling / "labels_with_tool_class.csv"

    # extract mill.zip file in folder_raw_data_milling if it hasn't been extracted yet
    if not (folder_raw_data_milling / "mill.mat").exists():
        print("Extracting mill.zip file...")
        with zipfile.ZipFile(folder_raw_data_milling / 'mill.zip', 'r') as zip_ref:
            zip_ref.extractall(folder_raw_data_milling)

    WINDOW_LEN = 64
    STRIDE = 64

    print("Creating dataframe from raw data...")
    milldata = MillingPrepMethodA(
        root = folder_raw_data,
        path_csv_labels = path_csv_labels,
        window_len=WINDOW_LEN,
        stride=STRIDE,
        download = True,
    )
    
    df = milldata.create_xy_dataframe()
    print("Shape of final df:", df.shape)

    # save the dataframe
    print("Saving dataframe...")
    df.to_csv(
        sub_folder_path / "milling.csv.gz",
        compression="gzip",
        index=False,
    )

    shutil.copy(
        Path(args.proj_dir) / "src/dataprep/make_dataset.py",
        sub_folder_path / "make_dataset.py",)


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
        "--sub_folder_name",
        type=str,
        default="milling_1",
        help="Name of the subfolder that will contain the extracted csv file",
    )


    args = parser.parse_args()

    main(args)
