import pandas as pd
from pathlib import Path
import argparse
import logging
import os
from multiprocessing import Pool
from src.models.filter import order_columns_on_results_df


def set_directories(args):

    scratch_path = Path.home() / "scratch"

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_model_dir:
        path_model_dir = Path(args.path_model_dir)
    else:
        if scratch_path.exists():
            print("Assume on HPC")
            path_model_dir = scratch_path / "feat-store" / "models"
        else:
            print("Assume on local compute")
            path_model_dir = proj_dir / "models"
    
    path_interim_dir = path_model_dir / args.interim_dir_name
    path_final_dir = path_model_dir / args.final_dir_name
    Path(path_final_dir).mkdir(parents=True, exist_ok=True)

    return proj_dir, path_model_dir, path_interim_dir, path_final_dir



def read_csv(filename):
    "converts a filename to a pandas dataframe"
    return pd.read_csv(filename)

def main(args):

    proj_dir, path_model_dir, path_interim_dir, path_final_dir = set_directories(args)

    # get a list of file names
    files = os.listdir(path_interim_dir)
    file_list = [
        path_interim_dir / filename for filename in files if filename.endswith(".csv")
    ]

    print("compiling csv files")

    # set up your pool
    with Pool(processes=args.n_cores) as pool:  # or whatever your hardware can support

        # have your pool map the file names to dataframes
        df = pool.map(read_csv, file_list)

        # reduce the list of dataframes to a single dataframe
        combined_df = pd.concat(df, ignore_index=True)

        return combined_df
 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build data sets for analysis")

    parser.add_argument(
        "--n_cores",
        type=int,
        default=1,
        help="Number of cores to use for multiprocessing",
    )

    parser.add_argument(
        "--final_dir_name",
        type=str,
        help="Folder name containing compiled csv.",
    )

    parser.add_argument(
        "--path_model_dir",
        dest="path_model_dir",
        type=str,
        help="Folder containing the trained model results",
    )

    parser.add_argument(
        "--compiled_csv_name",
        type=str,
        default="compiled_results.csv",
        help="The combined csv name.",
    )

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--interim_dir_name",
        type=str,
        help="Folder name containing all the interim result csv's that will be compiled into one.",
    )

    args = parser.parse_args()

    proj_dir, path_model_dir, path_interim_dir, path_final_dir = set_directories(args)

    df = main(args)
    df = order_columns_on_results_df(df)

    df.to_csv(path_final_dir / args.compiled_csv_name, index=False)
