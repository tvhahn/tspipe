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