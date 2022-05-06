import pandas as pd
from pathlib import Path
import argparse
import logging
import os
from multiprocessing import Pool


def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    interim_dir_name = args.interim_dir_name
    final_dir_name = args.final_dir_name
    
    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        print("Assume on HPC")

        path_interim_dir = scratch_path / "feat-store/models" / interim_dir_name
        path_final_dir = scratch_path / "feat-store/models" / final_dir_name
        Path(path_final_dir).mkdir(parents=True, exist_ok=True)

    else:
        print("Assume on local compute")
        path_interim_dir = proj_dir / "models" / interim_dir_name
        path_final_dir = proj_dir / "models" / final_dir_name
        Path(path_final_dir).mkdir(parents=True, exist_ok=True)

    return proj_dir, path_interim_dir, path_final_dir


def read_csv(filename):
    "converts a filename to a pandas dataframe"
    return pd.read_csv(filename)

def main(results_dir_path):

    # get a list of file names
    files = os.listdir(results_dir_path)
    file_list = [
        results_dir_path / filename for filename in files if filename.endswith(".csv")
    ]

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

    proj_dir, path_interim_dir, path_final_dir = set_directories(args)

    df = main(path_interim_dir)

    df.to_csv(path_final_dir / args.compiled_csv_name, index=False)
