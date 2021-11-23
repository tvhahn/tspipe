# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from numpy.core.fromnumeric import shape
from src.data.data_prep_utils import MillingDataPrep



def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    folder_raw_data_milling = project_dir / 'data/raw/milling'
    folder_processed_data_milling = project_dir / 'data/processed/milling'
    df_label_path = folder_processed_data_milling / "labels_with_tool_class.csv"

    milldata = MillingDataPrep(folder_raw_data_milling / 'mill.mat', path_df_labels=df_label_path, window_size=64, stride=64)
    df = milldata.create_xy_dataframe()
    print("Shape of final df:", df.shape)

    # save the dataframe
    df.to_csv(folder_processed_data_milling / "milling.csv.gz", compression="gzip", index=False)






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main(project_dir)
