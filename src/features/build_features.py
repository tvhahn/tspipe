from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import logging
from tsfresh import extract_features
from feat_param_dict import feat_dict

###############################################################################
# Set arguments 
###############################################################################

parser = argparse.ArgumentParser(description='Build features')

parser.add_argument('--path_data_folder', type=str, default='data/', help='Path to data folder that contains raw/interim/processed data folders')

parser.add_argument('--chunk-number', type=int, default=1, help='Chunk number')

args = parser.parse_args()

path_data_folder = Path(args.path_data_folder)
chunk_number = args.chunk_number # number of chunks to split dataframe into


###############################################################################
# Functions
###############################################################################

def milling_features(df_raw_milling, chunk_number):
    """Extracts features from the raw milling dataframe.
    
    Parameters
    ----------
    df_raw_milling : pandas.DataFrame
        Raw milling dataframe. Includes the "tool_class" and "time" columns.

    chunk_index : int
        Chunk index. Passed from the slurm script if on a HPC. Else, set to 0.
    
    """
    logger = logging.getLogger(__name__)
    logger.info("extracting features from raw milling dataframe")

    # check if "scratch" path exists in the home directory
    # if it does, assume we are on HPC
    scratch_path = Path.home() / "scratch"

    if scratch_path.exists():
        # get list of all cut_ids
        cut_id_list = list(df_raw_milling["cut_id"].unique())

    # extract features



def main(path_data_folder):
    """Runs feature engineering scripts to turn raw data from (../interim) into
    features ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    folder_raw_data_milling = project_dir / "data/raw/milling"
    folder_processed_data_milling = project_dir / "data/processed/milling"
    df_label_path = folder_processed_data_milling / "labels_with_tool_class.csv"

    milldata = MillingDataPrep(
        folder_raw_data_milling / "mill.mat",
        path_df_labels=df_label_path,
        window_size=64,
        stride=64,
    )
    
    df = milldata.create_xy_dataframe()
    print("Shape of final df:", df.shape)

    # save the dataframe
    df.to_csv(
        folder_processed_data_milling / "milling.csv.gz",
        compression="gzip",
        index=False,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)


