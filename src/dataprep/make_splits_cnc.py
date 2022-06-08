# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import argparse
import shutil
import pickle
import os
import pandas as pd
from multiprocessing import Pool
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import pickle
from scipy import stats
import tqdm
from tqdm.contrib.concurrent import process_map


def rename_cols_df(df):
    """Take a dataframe with various signals (e.g. current_main, current_sub, etc.)
    and re-lable the columns to standard names

    """

    # Dict of columns names to standard names
    col_name_change = {
        "Current_MainSpindle": "current_main",
        "Current_SubSpindle": "current_sub",
        "Power_MainSpindle": "power_main",
        "Power_SubSpindle": "power_sub",
        "CUT_Signal": "cut_signal",
        "Cut_Signal": "cut_signal",
        "Speed_SubSpindle": "speed_sub",
        "Speed_MainSpindle": "speed_main",
        "TOOL_Number": "tool_no",
        "Tool_Number": "tool_no",
        "INORM_MainSpindle": "current_main",
        "INORM_SubSpindle": "current_sub",
        "LMDAT_MainSpindle": "voltage_main",
        "LMDAT_SubSpindle": "voltage_sub",
        "INORM.1": "current_sub",
        "INORM": "current_main",
        "SPSPD": "speed_main",
        "SPSPD.1": "speed_sub",
        "SPEED": "speed_main",
        "SPEED.1": "speed_sub",
        "TCMD": "tcmd_z",
        "TCMD.1": "tcmd_x",
        "ERR": "error_z",
        "PMC": "pmc",
    }

    # Rename columns to standard names
    df.rename(columns=col_name_change, inplace=True)

    return df


def cut_signal_apply(cols):
    """Determine if the tool is in cut, or not, from PMC signal


    Explanation
    ===========
    The PMC signal is a binary number with 7 characters (e.g. 1111001, representing 121
    in base 10 number system). Another example: 0000110 represents 6. If the first digit
    in the sequence is a 1, then the tool is in cut (1000000 is equal to 64 in base 10).
    So if tool 6 is in cut, the PMC signal would be 1000110. 1000110 is equivalent to
    70 in base 10.

    So if the first digit is 1, the tool is in cut. The remaining digits equal the tool number.

    When the PMC signal is saved to the CSV, it is saved as a base 10 number. Work in the base 10 then.
    Subtract 64 from the number. If the result is greater than 0, then we know the tool is in cut, and the
    tool number is pmc_no - 64. If, after subtracting 64, the result is negative, we know that the tool
    is out of cut, and the tool number is equal to pmc_no.

    """
    pmc = cols[0]
    if (pmc - 64) > 0:
        return 1
    else:
        return 0


def tool_no_apply(cols):
    """Gets the tool number from the PMC signal

    Explanation
    ===========
    Same explanation as in the cut_signal_apply function

    """

    pmc = cols[0]
    if (pmc - 64) > 0:
        return int(pmc - 64)
    else:
        return int(pmc)


def extract_data_csv(file):
    """Extracts the useful columns from a csv file. The csv is a standard output from
    the Servoviewer application.

    Output is a pandas dataframe and the unix timestamp.

    Parameters
    ===========
    file : string
        The csv name that contains the cut info. e.g. auto$002.csv

    Returns
    ===========
    df : dataframe
        Pandas dataframe. Columns such as: current_main, current_sub, power_main,
        power_sub, cut_signal, speed_sub, tool_no, current_main, etc.

    unixtime : int
        Unix timestamp of when the cut started.
        See wikipedia for detail: https://en.wikipedia.org/wiki/Unix_time
        Example: 2018-10-23 08:45:55 --> 1540298755

    Future Work / Improvements
    ==========================
    - Rather than using pandas 'apply' function (which operates row-by-row),
    perform a vector operation (which operates across the entire vector at once,
    thus is much faster).

    """

    # get the unix timestamp for when the file was modified (http://bit.ly/2RW5cYo)
    unixtime = int(os.path.getmtime(file))

    # dictionary that contains the name of columns and their native format
    dict_int16_columns = {
        "speed_main": "int16",
        "speed_sub": "int16",
        "current_main": "int16",
        "current_sub": "int16",
        "cut_signal": "int16",
        "tool_no": "int16",
    }

    # load the csv
    # don't load certain rows as they contain meta info
    df = pd.read_csv((file), skiprows=[0, 1, 3], dtype="float32")

    df["cut_signal"] = df[["PMC"]].apply(cut_signal_apply, axis=1)  # get cut signal
    df["tool_no"] = df[["PMC"]].apply(tool_no_apply, axis=1)  # get tool_no
    df.drop(["Index", "Time", "PMC"], axis=1, inplace=True)  # remove the "index" column

    # standardize column names
    df = rename_cols_df(df)

    # cast some of the columns in their native datatype
    df = df.astype(dict_int16_columns)

    return (df, unixtime)


def extract_data_pickle(file):
    """Extracts the useful columns from a pickle file.

    Output is a pandas dataframe and the unix timestamp.

    The pickle file is often used when downsampling from 2000hz to 1000hz.

    Parameters
    ===========
    file : string
        The pickle file name containing the cut info. e.g. 1568213682.pickle

    Returns
    ===========
    df : dataframe
        Pandas dataframe. Columns such as: current_main, current_sub, power_main,
        power_sub, cut_signal, speed_sub, tool_no, current_main, etc.

    unixtime : int
        Unix timestamp of when the cut started.
        See wikipedia for detail: https://en.wikipedia.org/wiki/Unix_time
        Example: 2018-10-23 08:45:55 --> 1540298755

    Future Work / Improvements
    ==========================
    - Rather than using pandas 'apply' function (which operates row-by-row),
    perform a vector operation (which operates across the entire vector at once,
    thus is much faster).

    """

    # dictionary that contains the name of columns and their native format
    dict_int16_columns = {
        "speed_main": "int16",
        "speed_sub": "int16",
        "cut_signal": "int16",
        "tool_no": "int16",
    }

    # load the pickle file
    with open(file, "rb") as input_file:
        d = pickle.load(input_file)

    unixtime = list(d.keys())[0]  # get the unix timestamp of the cut
    df = d[unixtime][0]
    df = df.astype(dtype="float32")  # cast columns as float32
    df = rename_cols_df(df)  # standardize column names
    try:
        df.drop(["time"], axis=1, inplace=True)  # remove the "index" column
    except:
        pass

    # cast some of the columns in their native datatype
    try:
        df = df.astype(dict_int16_columns)
    except:
        dict_int16_columns = {"cut_signal": "int16", "tool_no": "int16"}

        df = df.astype(dict_int16_columns)

    return (df, unixtime)


def extract_data_mat(m):
    """Extracts the useful columns from a matlab file.

    Output is a pandas dataframe and the unix timestamp.

    Parameters
    ===========
    m : dict
        Takes a dictionary created by the scipy.io.loadmat function
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

    Returns
    ===========
    df : dataframe
        Pandas dataframe. Columns: current_main, current_sub, power_main,
        power_sub, cut_signal, speed_sub, tool_no, current_main,
        voltage_main, voltage_sub

    unixtime : int
        Unix timestamp of when the cut started.
        See wikipedia for detail: https://en.wikipedia.org/wiki/Unix_time
        Example: 2018-10-23 08:45:55 --> 1540298755

    """

    # Record the timestamp. Use try as two different formats
    try:
        time_stamp = m["TimeStamp"]
    except:
        pass

    try:
        time_stamp = m["Date_Time"]
    except:
        pass

    # Convert the timestamp to unix-time and date-time formats
    for x in time_stamp:
        t = np.array2string(x)

        # Convert string to datetime http://bit.ly/2WGCZnL
        d = datetime.strptime(t, "'%d-%b-%Y %H:%M:%S'")

        # Convert to unix time http://bit.ly/2GJMf50
        unixtime = int(time.mktime(d.timetuple()))

    # Tuple of columns to remove
    entries_remove = (
        "__header__",
        "__version__",
        "__globals__",
        "TimeStamp",
        "Date_Time",
        "Index",
        "Time",
        "Tool_Offset_Data",
        "Pulsecode_POsition_X",
        "Pulsecode_POsition_Z",
        "PhaseA_SubSpindle",
        "PhaseA_MainSpindle",
    )

    # Dict of columns names to standard names
    col_name_change = {
        "Current_MainSpindle": "current_main",
        "Current_SubSpindle": "current_sub",
        "Power_MainSpindle": "power_main",
        "Power_SubSpindle": "power_sub",
        "CUT_Signal": "cut_signal",
        "Cut_Signal": "cut_signal",
        "Speed_SubSpindle": "speed_sub",
        "Speed_MainSpindle": "speed_main",
        "TOOL_Number": "tool_no",
        "Tool_Number": "tool_no",
        "INORM_MainSpindle": "current_main",
        "INORM_SubSpindle": "current_sub",
        "LMDAT_MainSpindle": "voltage_main",
        "LMDAT_SubSpindle": "voltage_sub",
        "TorqueCommand_Z": "tcmd_z",
        "Psition_ERR_Z": "error_z",
        "Psition_ERR_X": "error_x",
    }

    # Remove the unnecessary columns from the data-set
    # (stackoverflow link: https://bit.ly/2D8wcNU)
    for k in entries_remove:
        m.pop(k, None)

    # Create a list of the column names
    index_list = []
    for k in m:
        index_list.append(k)

    # Create a dummy pandas df to instantiate the dataframe
    # could probably fix this up in later versions...
    df = pd.DataFrame(np.reshape(m[index_list[1]], -1), columns=["dummy"])

    # Iterate through the variables and append onto our 'dummy' df
    for k in m:
        df[k] = pd.DataFrame(np.reshape(m[k], -1))
    df = df.drop(["dummy"], axis=1)  # Drop the 'dummy' column

    # Rename columns to standard names
    df.rename(columns=col_name_change, inplace=True)

    # dictionary that contains the native data types of certain columns
    dict_int16_columns = {
        "speed_main": "int16",
        "speed_sub": "int16",
        "current_main": "int16",
        "current_sub": "int16",
        "cut_signal": "int16",
        "tool_no": "int16",
    }

    # cast all columns as float32 (as opposed to float64)
    df = df.astype("float32")

    # cast some of the columns in their native datatype
    try:
        df = df.astype(dict_int16_columns)
    except:
        dict_int16_columns = {
            "current_main": "int16",
            "current_sub": "int16",
            # "cut_signal": "int16",
            "tool_no": "int16",
        }
        df = df.astype(dict_int16_columns)

    return (df, unixtime)


def stable_speed_section(df):
    """Identifies the stable speed region in a cut. Returns a dataframe with
    only the stable speed region.

    The function takes a standard cut dataframe. I then finds the most common
    speed value (by using the numpy mode function). It then windows off the
    cut between the first and last instance of the mode.


    Parameters
    ===========
    df : pandas dataframe
        A standard cut dataframe that includes both sub and main spindle speeds


    Returns
    ===========
    df : pandas dataframe
        Dataframe with only the region that is identified as "stable speed"


    Future Work / Improvements
    ==========================
    -Rather than using "speed" to check which spindle is most active,
    should we use something like current instead?

    """

    def find_stable_speed(df, speed_array):

        # get absolute value of the speed array
        speed_array = np.abs(speed_array)

        # find the most common speed value (the mode)
        mode_speed = stats.mode(speed_array)[0][0]

        # find the index of the most commons speed value
        percentage_val = 0.005
        l = np.where(
            (speed_array > (mode_speed - mode_speed * percentage_val))
            & (speed_array < (mode_speed + mode_speed * percentage_val))
        )[0]

        # now create the dataframe that only includes the range
        # of indices where the most common speed values are
        df = df[l[0] : l[-1]]

        return df

    df = df.reset_index()

    # check to see if the sub or main spindle is the one most active
    if np.abs(df["speed_sub"].mean()) > np.abs(df["speed_main"].mean()):
        speed_array = df["speed_sub"].to_numpy()
        df = find_stable_speed(df, speed_array)

    else:
        speed_array = df["speed_main"].to_numpy()
        df = find_stable_speed(df, speed_array)

    return df


def split_df(df, cut_time, stable_speed_only=False):
    """Load a dataframe of a cut and split it up by tool # and cut_signal==true

    Parameters
    ===========
    df : pandas dataframe
        Dataframe of a single cut

    cut_time : int
        Unix timestamp of the cut. Will be converted to a string.

    Returns
    ===========
    dict_cuts : dictionary
        Dictionary of the split cuts cut -- labeled by timestamp, tool number, and sequence
        e.g. {1548788710_22_0: ...data... , 1548788710_22_1: ...data... }

    """

    # print("Starting split for:\t", str(cut_time))

    # use np.split to split dataframe based on condition: http://bit.ly/2GEFBwr
    split_data = np.split(df, *np.where(df.cut_signal == 0))

    # empty dictionary to store the individual tool cuts and index
    dict_cuts = {}
    tool_index = {}

    for x in split_data:

        if len(x) > 1:

            # we dynamically create the tool_index
            # if the tool is not in the "tool_index", then we know
            # that it is at index 0
            try:
                index = tool_index[x.tool_no.iloc[1]]
            except:
                tool_index[x.tool_no.iloc[1]] = 0
                index = 0

            name = str(cut_time) + "_{}_{}".format(x.tool_no.iloc[1], index)

            if stable_speed_only == True:
                dict_cuts[str(name)] = stable_speed_section(x.iloc[1:])
                tool_index[x.tool_no.iloc[1]] += 1
            else:
                dict_cuts[str(name)] = x.iloc[1:]
                tool_index[x.tool_no.iloc[1]] += 1
        else:
            pass

    # print("Finished split for:\t", str(cut_time))
    return dict_cuts


def load_cut_save_split(file_dict):
    file_path = file_dict["file_path"]
    unix_date = file_dict["unix_date"]
    save_dir = file_dict["save_dir"]

    if file_path.suffix == ".mat":
        df, _ = extract_data_mat(sio.loadmat(file_path))
    elif file_path.suffix == ".pickle":
        df, _ = extract_data_pickle(file_path)
    elif file_path.suffix == ".csv":
        df, _ = extract_data_csv(file_path)
    else:
        print("File type not recognized for file:", file_path)

    dict_cuts = split_df(df, unix_date)

    for k, v in dict_cuts.items():
        pickle_out = open(save_dir / f"{k}.pickle", "wb")
        pickle.dump(v, pickle_out)
        pickle_out.close()


def main(args):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    path_data_dir = Path(args.path_data_dir)
    print("path_data_dir:", path_data_dir)

    if args.save_dir_name is None:
        save_dir = path_data_dir / "raw/cnc/splits"
    else:
        save_dir = path_data_dir / "raw/cnc" / args.save_dir_name

    print("save_dir:", save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    df_labels = pd.read_csv(
        path_data_dir
        / "processed/cnc/high_level_labels_MASTER_update2020-08-06_new-jan-may-data.csv"
    )

    file_list = [
        {
            "file_path": path_data_dir
            / "raw/cnc"
            / "/".join(Path(cut_dir).parts[7:])
            / file_name,
            "unix_date": unix_date,
            "save_dir": save_dir,
        }
        for cut_dir, file_name, unix_date in zip(
            df_labels.cut_dir, df_labels.file_name, df_labels.unix_date
        )
    ]

    # set up your pool
    with Pool(processes=args.n_cores) as pool:  # or whatever your hardware can support
        r = list(
            tqdm.tqdm(pool.imap(load_cut_save_split, file_list), total=len(file_list))
        )

        # pool.map(load_cut_save_split, file_list)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="Create dataframe from raw data")

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
        default="data/",
        help="Path to data folder that contains raw/interim/processed data folders",
    )

    parser.add_argument(
        "--n_cores",
        type=int,
        default=2,
        help="Number of processes to use for multiprocessing",
    )

    parser.add_argument(
        "--save_dir_name",
        type=str,
        help="Name of the folder where the saved cut files will be stored",
    )

    args = parser.parse_args()

    main(args)
