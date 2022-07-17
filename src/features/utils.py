from pathlib import Path
import pandas as pd
import argparse
import logging
from tsfresh import extract_features
from feat_param_dict import comprehensive_features
import json


def load_feat_json(path_feat_json):
    """Loads the feat_dict json file.
    Used to create the feature dictionary.

    Parameters
    ----------
    path_feat_json : Path
        Path to the feat_dict json file.

    Returns
    -------
    feat_dict : dict
        Dictionary of feature extraction parameters.

    """
    logger = logging.getLogger(__name__)
    logger.info("loading feat_dict json file")

    with open(path_feat_json, "r") as f:
        feat_dictionary = json.load(f)

    return feat_dictionary


def set_directories(args):
    """Sets the directories for the raw, interim, and processed data."""

    assert args.dataset == "cnc" or args.dataset == "milling", "Dataset must be either 'cnc' or 'milling'"


    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    path_raw_dir = path_data_dir / "raw" / args.dataset / args.raw_dir_name

    path_processed_dir = path_data_dir / "processed" / args.dataset / args.processed_dir_name
    path_processed_dir.mkdir(parents=True, exist_ok=True)

    path_interim_dir = path_data_dir / "interim" / args.dataset / args.interim_dir_name
    path_interim_dir.mkdir(parents=True, exist_ok=True)

    return proj_dir, path_data_dir, path_raw_dir, path_interim_dir, path_processed_dir
