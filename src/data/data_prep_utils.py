from pathlib import Path
import scipy.io as sio
import numpy as np
import pandas as pd


class MillingDataPrep:
    def __init__(self, path_raw_data, path_df_labels=None, window_size=64, stride=64, cut_drop_list=[17, 94]):
        """Prepare the UC Berkeley Milling dataset for training.

        Parameters
        ----------
        path_raw_data : pathlib
            Path to the raw data folder. Should point to a 'mill.mat' or similar file.

        path_df_labels : pathlib, optional
            Path to the dataframe with the labels. If not provided, the dataframe must be created.

        window_size : int
            Size of the window to be used for the sliding window.

        stride : int
            Size of the stride to be used for the sliding window.

        cut_drop_list : list
            List of cut numbers to be dropped from the dataset. cut_no 17 and 94 are erroneous.
        
        """

        self.data_file = path_raw_data # path to the raw data file
        self.window_size = window_size # size of the window
        self.stride = stride # stride between windows

        if path_df_labels is None:
            print("Warning: no csv defined for creating labels")
        else:
            self.df_labels = pd.read_csv(path_df_labels) # path to the labels file with tool class
            self.df_labels.drop(cut_drop_list, inplace=True) # drop the cuts that are bad
            self.df_labels.reset_index(drop=True, inplace=True) # reset the index

        # load the data from the matlab file
        m = sio.loadmat(self.data_file, struct_as_record=True)

        # store the 'mill' data in a seperate np array
        self.data = m["mill"]

        self.field_names = self.data.dtype.names
        self.signal_names = self.field_names[7:][::-1]

    def create_labels(self):
        """Function that will create the label dataframe from the mill data set
        
        Only needed if the dataframe with the labels is not provided.
        """

        # create empty dataframe for the labels
        df_labels = pd.DataFrame()

        # get the labels from the original .mat file and put in dataframe
        for i in range(7):
            # list for storing the label data for each field
            x = []

            # iterate through each of the unique cuts
            for j in range(167):
                x.append(self.data[0, j][i][0][0])
            x = np.array(x)
            df_labels[str(i)] = x

        # add column names to the dataframe
        df_labels.columns = self.field_names[0:7]

        # create a column with the unique cut number
        df_labels["cut_no"] = [i for i in range(167)]

        def tool_state(cols):
            """Add the label to the cut. 
            
            Categories are:
            Healthy Sate (label=0): 0~0.2mm flank wear
            Degredation State (label=1): 0.2~0.7mm flank wear
            Failure State (label=2): >0.7mm flank wear 
            """
            # pass in the tool wear, VB, column
            vb = cols

            if vb < 0.2:
                return 0
            elif vb >= 0.2 and vb < 0.7:
                return 1
            elif pd.isnull(vb):
                pass
            else:
                return 2

        # apply the label to the dataframe
        df_labels["tool_class"] = df_labels["VB"].apply(tool_state)

        return df_labels

    def create_data_array(self, cut_no):
        """Create an array from a cut sample.

        Parameters
        ===========
        cut_no : int
            Index of the cut to be used.

        Returns
        ===========
        sub_cut_array : np.array
            Array of the cut samples. Shape of [n0. samples, sample len, features/sample]

        sub_cut_labels : np.array
            Array of the labels for the cut samples. Shape of [# samples, # features/sample]

        """
        
        assert cut_no in self.df_labels["cut_no"].values, "Cut number must be in the dataframe"

        # create a numpy array of the cut 
        # with a final array shape like [no. cuts, len cuts, no. signals]
        cut = self.data[0, cut_no]
        for i, signal_name in enumerate(self.signal_names):
            if i == 0:
                cut_array = cut[signal_name].reshape((9000, 1))
            else:
                cut_array = np.concatenate((cut_array, cut[signal_name].reshape((9000, 1))), axis=1)

        # select the start and end of the cut
        start = self.df_labels[self.df_labels["cut_no"] == cut_no]["window_start"].values[0]
        end = self.df_labels[self.df_labels["cut_no"] == cut_no]["window_end"].values[0]
        cut_array = cut_array[start:end,:]

        # instantiate the "temporary" list to store the sub-cuts and metadata
        sub_cut_list = []
        sub_cut_id_list = []
        sub_cut_label_list = []

        # get the labels for the cut
        label = self.df_labels[self.df_labels["cut_no"] == cut_no]["tool_class"].values[0]

        # fit the strided windows into the dummy_array until the length
        # of the window does not equal the proper length (better way to do this???)
        for i in range(cut_array.shape[0]):
            windowed_signal = cut_array[i * self.stride : i * self.stride + self.window_size]

            # if the windowed signal is the proper length, add it to the list
            if windowed_signal.shape == (self.window_size, 6):
                sub_cut_list.append(windowed_signal)

                # create sub_cut_id fstring to keep track of the cut_id and the window_id
                sub_cut_id_list.append(f"{cut_no}_{i}")

                # create the sub_cut_label and append it to the list
                sub_cut_label_list.append(int(label))

            else:
                break

        sub_cut_array = np.array(sub_cut_list)

        sub_cut_ids = np.expand_dims(np.array(sub_cut_id_list, dtype=str), axis=1)
        sub_cut_ids = np.repeat(sub_cut_ids, sub_cut_array.shape[1], axis=1)

        sub_cut_labels = np.expand_dims(np.array(sub_cut_label_list, dtype=int), axis=1)
        sub_cut_labels = np.repeat(sub_cut_labels, sub_cut_array.shape[1], axis=1)

        # take the length of the signals in the sub_cut_array
        # and divide it by the frequency (250 Hz) to get the time (seconds) of each sub-cut
        sub_cut_times = np.expand_dims(np.arange(0, sub_cut_array.shape[1])/250.0, axis=0)
        sub_cut_times = np.repeat(sub_cut_times, sub_cut_array.shape[0], axis = 0,)
        
        sub_cut_labels_ids_times = np.stack((sub_cut_labels, sub_cut_ids, sub_cut_times), axis=2)

        return sub_cut_array, sub_cut_labels, sub_cut_ids, sub_cut_times, sub_cut_labels_ids_times

    def create_xy_arrays(self):
        """Create the x and y arrays used in deep learning.

        Returns
        ===========
        x_array : np.array
            Array of the cut samples. Shape of [no. samples, sample len, features/sample]

        y_array : np.array
            Array of the labels for the cut samples. Shape of [no. samples, sample len, label/ids/times]
        
        """

        # create a list to store the x and y arrays
        x = []  # instantiate X's
        y_labels_ids_times = []  # instantiate y's
        

        # iterate throught the df_labels
        for i in self.df_labels.itertuples():
            (sub_cut_array, sub_cut_labels, sub_cut_ids, 
            sub_cut_times, sub_cut_labels_ids_times) = self.create_data_array(i.cut_no)
        
            x.append(sub_cut_array)
            y_labels_ids_times.append(sub_cut_labels_ids_times)

        return np.vstack(x), np.vstack(y_labels_ids_times)

    def create_xy_dataframe(self):
        """Create a flat dataframe (2D array) of the x and y arrays.

        Returns
        ===========
        df : pd.DataFrame
            Single flat dataframe containing each sample and its labels.
        
        """

        x, y_labels_ids_times = self.create_xy_arrays() # create the x and y arrays

        # concatenate the x and y arrays and reshape them to be a flat array (2D)
        x_labels = np.reshape(np.concatenate((x, y_labels_ids_times), axis=2),(-1, 9))

        # define the column names and the data types
        col_names = [s.lower() for s in list(self.signal_names)] + ["tool_class", "cut_id", "time"] 
        col_dtype = [np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, int, str, np.float32]
        col_dtype_dict = dict(zip(col_names, col_dtype))
        col_names_ordered = ['cut_id', 'case', 'time', 'ae_spindle', 'ae_table', 'vib_spindle', 'vib_table', 'smcdc', 'smcac','tool_class']

        # create a dataframe from the x and y arrays
        df = pd.DataFrame(x_labels, columns=col_names, dtype=str).astype(col_dtype_dict)
        df["case"] = df["cut_id"].str.split("_").str[0] # split the cut_id by "_" and take the first element (case)
        df = df[col_names_ordered] # reorder the columns
                
        return df