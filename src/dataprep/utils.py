from pathlib import Path
import scipy.io as sio
import numpy as np
import pandas as pd

"""
Generic functions used in the dataprep module.
"""

def set_directories(args):
    """Sets the directories for the raw, interim, and processed data."""

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    path_raw_dir = path_data_dir / "raw"
    path_raw_dir.mkdir(parents=True, exist_ok=True)

    return proj_dir, path_data_dir, path_raw_dir





    
