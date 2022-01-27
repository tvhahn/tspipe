from pathlib import Path
import pandas as pd
import argparse
import logging
from feat_param_dict import comprehensive_features, dummy_features
from src.features.utils import load_feat_json
from src.features.datasets.milling_utils import milling_features

###############################################################################
# Functions
###############################################################################
    

def main(path_data_folder):
    """Runs feature engineering scripts to turn raw data from (../interim) into
    features ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final features set from raw data")

    ### Milling data ###
    folder_raw_data_milling = path_data_folder / "raw/milling"
    folder_interim_data_milling = path_data_folder / "interim/milling"
    folder_processed_data_milling = path_data_folder / "processed/milling"

    # if interim & processed folders don't exist, create them
    Path(folder_interim_data_milling).mkdir(parents=True, exist_ok=True)
    Path(folder_processed_data_milling).mkdir(parents=True, exist_ok=True)

    # read in raw milling data to a pandas dataframe
    df = pd.read_csv(folder_raw_data_milling / "milling.csv.gz", compression='gzip',)
    print("df.columns", df.columns)
    
    print("Shape of df:", df.shape)


    # load feat_dict json file if the argument is passed
    # else the default feat_dict will be used (from feat_param_dict.py)
    if args.path_feat_json:
        feat_dict = load_feat_json(args.path_feat_json)
    else:
        # feat_dict = comprehensive_features
        feat_dict = dummy_features

    df_feat = milling_features(df, n_chunks, chunk_index, feat_dict)

    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        # save the dataframe oh HPC
        df_feat.to_csv(
            folder_interim_data_milling / f"milling_{chunk_index}.csv",
            index=False,
        )
    else:
        # save the dataframe on local machine
        df_feat.to_csv(
            folder_processed_data_milling / "milling.csv", index=False,
        )

    # assert that tool_class column exists
    assert "tool_class" in df_feat.columns, "tool_class column does not exist"


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    ### parse arguments

    parser = argparse.ArgumentParser(description="Build features")

    parser.add_argument(
        "--path_data_folder",
        type=str,
        default="data/",
        help="Path to data folder that contains raw/interim/processed data folders",
    )

    parser.add_argument(
        "--path_feat_json",
        type=str,
        help="Path to feat_dict.json file",
    )

    parser.add_argument(
        "--n_chunks", type=int, default=1, help="Number of chunks to split dataframe into"
    )

    parser.add_argument(
        "--chunk_index",
        type=int,
        default=0,
        help="Chunk index passed from the slurm script",
    )

    args = parser.parse_args()

    path_data_folder = Path(args.path_data_folder)
    n_chunks = int(args.n_chunks)  # number of chunks to split dataframe into
    chunk_index = int(args.chunk_index) - 1

    main(path_data_folder)
