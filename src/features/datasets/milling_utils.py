from pathlib import Path
import logging
from tsfresh import extract_features
from feat_param_dict import comprehensive_features


def milling_features(df, n_chunks, chunk_index, feature_dictionary=comprehensive_features):
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

    df_raw_labels = df[['cut_id', 'cut_no', 'case', 'tool_class']].drop_duplicates().copy()
    df = df.drop(columns=['cut_no', 'case', 'tool_class'])

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

    else:
        # extract features on local machine
        df_feat = extract_features(
            df,
            column_id="cut_id",
            column_sort="time",
            default_fc_parameters=feature_dictionary,
            disable_progressbar=False,
        )

    # return the dataframe with the features and the labels
    return df_feat.reset_index().rename(columns={'index':'cut_id'}).merge(df_raw_labels, on='cut_id', how='left')