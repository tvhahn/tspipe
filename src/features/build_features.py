from pathlib import Path
import pandas as pd
import argparse
import logging
from tsfresh import extract_features
from feat_param_dict import feat_dict


###############################################################################
# Functions
###############################################################################


def milling_features(df, n_chunks, chunk_index, feature_dictionary=feat_dict):
    """Extracts features from the raw milling dataframe.

    Parameters
    ----------
    df_raw_milling : pandas.DataFrame
        Raw milling dataframe. Includes the "tool_class" and "time" columns.

    n_chunks : int
        Chunk index. Passed from the slurm script if on a HPC. Else, set to 0.

    """
    logger = logging.getLogger(__name__)
    logger.info("extracting features from raw milling dataframe")

    # check if "scratch" path exists in the home directory
    # if it does, assume we are on HPC
    scratch_path = Path.home() / "scratch"

    df_raw_labels = df[['cut_id', 'case', 'tool_class']].drop_duplicates().copy()
    df = df.drop(columns=['case', 'tool_class'])

    if scratch_path.exists():
        # get list of all cut_ids
        cut_id_list = list(df["cut_id"].unique())

        # create a list of the cut_ids to be processed in each chunk
        n_samples_per_chunk = int(len(cut_id_list) / n_chunks)
        cut_id_list_chunks = [
            cut_id_list[i : i + n_samples_per_chunk]
            for i in range(0, len(cut_id_list), n_samples_per_chunk)
        ]

        # extract features on HPC
        df_feat = extract_features(
            df[df["cut_id"].isin(cut_id_list_chunks[chunk_index])],
            column_id="cut_id",
            column_sort="time",
            default_fc_parameters=feature_dictionary,
            disable_progressbar=False,
        )

        df_feat = df_feat.reset_index().rename(columns={'index':'cut_id'})
        df_feat.merge(df_raw_labels, on='cut_id', how='left')

    else:
        # extract features on local machine
        df_feat = extract_features(
            df,
            column_id="cut_id",
            column_sort="time",
            default_fc_parameters=feature_dictionary,
            disable_progressbar=False,
        )

        df_feat = df_feat.reset_index().rename(columns={'index':'cut_id'})
        df_feat.merge(df_raw_labels, on='cut_id', how='left')

    return df_feat

    

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

    # for testing purposes only include some cuts
    # df = df[df["cut_id"].isin(['0_0', '0_1', '0_2'])]

    df_feat = milling_features(df, n_chunks, chunk_index)

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

    print(df_feat["tool_class"][0:3])


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
