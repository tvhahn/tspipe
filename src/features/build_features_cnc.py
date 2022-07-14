from pathlib import Path
import pandas as pd
import argparse
import logging
from feat_param_dict import comprehensive_features, dummy_features
from src.features.utils import load_feat_json
from tsfresh import extract_features


def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    path_raw_dir = path_data_dir / "raw" / "cnc" / args.raw_dir_name

    path_processed_dir = path_data_dir / "processed" / "cnc" / args.processed_dir_name
    path_processed_dir.mkdir(parents=True, exist_ok=True)

    return proj_dir, path_data_dir, path_raw_dir, path_processed_dir

    
def cnc_features(df, n_chunks, chunk_index, feature_dictionary=comprehensive_features):
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

    if scratch_path.exists():
        # get list of all cut_ids
        id_list = list(df["id"].unique())

        # create a list of the cut_ids to be processed in each chunk
        n_samples_per_chunk = int(len(id_list) / n_chunks)
        id_list_chunks = [
            id_list[i : i + n_samples_per_chunk]
            for i in range(0, len(id_list), n_samples_per_chunk)
        ]

        # extract features on HPC
        df_feat = extract_features(
            df[df["id"].isin(id_list_chunks[chunk_index])],
            column_id="id",
            column_sort="time",
            default_fc_parameters=feature_dictionary,
            disable_progressbar=False,
        )

    else:
        # extract features on local machine
        df_feat = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=feature_dictionary,
            disable_progressbar=False,
        )

    # return the dataframe with the features and the labels
    return df_feat.reset_index().rename(columns={'index':'id'})


def main(args):
    """Runs feature engineering scripts to turn raw data from (../interim) into
    features ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final features set from raw data")


    proj_dir, path_data_dir, path_raw_dir, path_processed_dir = set_directories(args)



    # read in raw milling data to a pandas dataframe
    df = pd.read_csv(path_raw_dir / args.raw_file_name,)
    print("df.columns", df.columns)
    
    print("Shape of df:", df.shape)


    # load feat_dict json file if the argument is passed
    # else the default feat_dict will be used (from feat_param_dict.py)
    if args.path_feat_json:
        feat_dict = load_feat_json(args.path_feat_json)
    else:
        # feat_dict = comprehensive_features
        print("Using dummy features")
        feat_dict = dummy_features

    df_feat = cnc_features(df, n_chunks, chunk_index, feat_dict)

    # assert "tool_class" in df_feat.columns, "tool_class column does not exist"
    # assert "cut_id" in df_feat.columns, "tool_class column does not exist"

    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        # save the dataframe oh HPC
        df_feat.to_csv(
            path_processed_dir / f"cnc_{chunk_index}.csv",
            index=False,
        )
    else:
        # save the dataframe on local machine
        df_feat.to_csv(
            path_processed_dir / args.feat_file_name, index=False,
        )




if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    ### parse arguments

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
        "--path_feat_json",
        type=str,
        help="Path to feat_dict.json file",
    )

    parser.add_argument(
        "--n_chunks", type=int, default=1, help="Number of chunks to split dataframe into"
    )

    parser.add_argument(
        "--raw_dir_name",
        type=str,
        default="data_raw_processed",
        help="Name of the subfolder that contains the final raw csv file",
    )

    parser.add_argument(
        "--raw_file_name",
        type=str,
        help="Name of the raw csv file that the features will be made from.",
    )

    parser.add_argument(
        "--feat_file_name",
        type=str,
        help="Name of the final feature file.",
    )

    # parser.add_argument(
    #     "--cut_ids_file_name",
    #     type=str,
    #     help="Name of the csv that contains the list of all the unique cut ids",
    # )

    parser.add_argument(
        "--processed_dir_name",
        default="cnc_features",
        type=str,
        help="Name of the save directory. Used to store features. Located in data/processed/cnc",
    )

    parser.add_argument(
        "--high_level_label_file_name",
        type=str,
        help="Name of the high level label file. This file includes the meta data and labels for each unique id",
    )

    parser.add_argument(
        "--chunk_index",
        type=int,
        default=0,
        help="Chunk index passed from the slurm script",
    )

    args = parser.parse_args()

    # path_data_folder = Path(args.path_data_folder)
    n_chunks = int(args.n_chunks)  # number of chunks to split dataframe into
    chunk_index = int(args.chunk_index) - 1

    main(args)