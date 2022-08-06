import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import seaborn as sns
import pandas as pd
import argparse
from src.models.utils import cnc_add_y_label_binary
from datetime import datetime
from pathlib import Path

###############################################################################
# Generic plotting functions
###############################################################################


def segment_line_by_sliding_window(x, y, x_axis_n=1000):

    y_adj = []  # y values, adjusted
    x_adj = []  # x values, adjusted

    # break the x-axis up by moving a sliding window along it
    for j, i in enumerate(sliding_window_view(np.linspace(0, 1, x_axis_n), 2)):
        window_segment = np.where((x >= i[0]) & (x <= i[1]))[0]

        # if there are no values in the window, then
        # use the last precision value in the window
        if len(window_segment) == 0:
            x_adj.append(np.mean(i))
            y_adj.append(y_adj[-1])
        else:
            x_adj.append(np.mean(i))
            y_adj.append(np.mean(y[window_segment]))

    return np.array(x_adj), np.array(y_adj)


def plot_pr_roc_curves_kfolds(
    precision_array,
    recall_array,
    fpr_array,
    tpr_array,
    rocauc_array,
    prauc_array,
    percent_anomalies_truth=0.073,
    path_save_dir=None,
    save_name="model_curves",
    save_plot=False,
    dpi=300,
):
    """
    Plot the precision-recall curves and the ROC curves for the different k-folds used in
    cross-validation. Also show the average PR and ROC curves.

    :param precision_array: array of precision values for each k-fold
    :param recall_array: array of recall values for each k-fold
    :param fpr_array: array of false positive rate values for each k-fold
    :param tpr_array: array of true positive rate values for each k-fold
    :param rocauc_array: array of ROC AUC values for each k-fold
    :param prauc_array: array of PR AUC values for each k-fold
    :param percent_anomalies_truth: the percentage of anomalies in the dataset

    """

    # sns whitegrid context
    sns.set(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, dpi=150)
    fig.tight_layout(pad=5.0)

    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)

    ######
    # plot the precision recall curves
    precisions_all_segmented = []
    for p, r in zip(precision_array, recall_array):
        r_adj, p_adj = segment_line_by_sliding_window(r, p, x_axis_n=10000)
        precisions_all_segmented.append(p_adj)

    precisions_all_segmented = np.array(precisions_all_segmented)

    for i, (p, r) in enumerate(zip(precision_array, recall_array)):
        if i == np.shape(precision_array)[0] - 1:
            axes[0].plot(
                r[:], p[:], label="k-fold model", color="gray", alpha=0.5, linewidth=1
            )
        else:
            axes[0].plot(r[:], p[:], color="grey", alpha=0.5, linewidth=1)

    axes[0].plot(
        r_adj,
        precisions_all_segmented.mean(axis=0),
        label="Average model",
        color=pal[5],
        linewidth=2,
    )

    axes[0].plot(
        np.array([0, 1]),
        np.array([percent_anomalies_truth, percent_anomalies_truth]),
        marker="",
        linestyle="--",
        label="No skill model",
        color="orange",
        linewidth=2,
        zorder=0,
    )

    axes[0].legend()
    axes[0].title.set_text("Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].text(
        x=-0.05,
        y=-0.3,
        s=f"PR AUC: {prauc_array.mean():.3f} (avg), {prauc_array.min():.3f} (min), {prauc_array.max():.3f} (max)",
        # fontsize=10,
        horizontalalignment="left",
        verticalalignment="center",
        rotation="horizontal",
        alpha=1,
    )

    ######
    # plot the ROC curves
    roc_all_segmented = []
    for t, f in zip(tpr_array, fpr_array):
        f_adj, t_adj = segment_line_by_sliding_window(f, t, x_axis_n=10000)
        roc_all_segmented.append(t_adj)

    roc_all_segmented = np.array(roc_all_segmented)

    for i, (t, f) in enumerate(zip(tpr_array, fpr_array)):
        if i == np.shape(tpr_array)[0] - 1:
            axes[1].plot(
                f[:], t[:], label="k-fold models", color="gray", alpha=0.5, linewidth=1
            )
        else:
            axes[1].plot(f[:], t[:], color="grey", alpha=0.5, linewidth=1)

    axes[1].plot(
        f_adj,
        roc_all_segmented.mean(axis=0),
        label="Average of k-folds",
        color=pal[5],
        linewidth=2,
    )

    axes[1].plot(
        np.array([0, 1]),
        np.array([0, 1]),
        marker="",
        linestyle="--",
        label="No skill",
        color="orange",
        linewidth=2,
        zorder=0,
    )

    axes[1].title.set_text("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].text(
        x=-0.05,
        y=-0.3,
        s=f"ROC AUC: {rocauc_array.mean():.3f} (avg), {rocauc_array.min():.3f} (min), {rocauc_array.max():.3f} (max)",
        # fontsize=10,
        horizontalalignment="left",
        verticalalignment="center",
        rotation="horizontal",
        alpha=1,
    )

    for ax in axes.flatten():
        ax.yaxis.set_tick_params(labelleft=True, which="major")
        ax.grid(False)

    if save_plot:
        if path_save_dir is None:
            path_save_dir = "./"

        # save as both png and pdf
        plt.savefig(path_save_dir / f"{save_name}.png", dpi=dpi, bbox_inches="tight")
        plt.savefig(path_save_dir / f"{save_name}.pdf", bbox_inches="tight")
        plt.cla()
        plt.close()
    else:
        plt.show()


def plot_lollipop_results(
    df,
    metric="prauc",
    plt_title=None,
    path_save_dir=None,
    save_name="results_lollipop",
    save_plot=False,
    dpi=300,
):
    """Plot the top performing models and by a metric.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the results of the models.
    metric : str
        The metric to plot the results by.
    plt_title : str
        The title of the plot.
    path_save_name : Path
        The name (as a path object) of the file to save the plot to.
    save_plot : bool
        Whether to save the plot or not. Otherwise, it will be shown.
    dpi : int
        The resolution of the plot if saved as png or if shown.
    """

    df = df.sort_values(by=[f"{metric}_avg"], ascending=True).reset_index(drop=True)

    plt.style.use("seaborn-whitegrid")  # set style because it looks nice
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 12),
    )

    # color palette to choose from
    darkblue = "#1f253f"
    lightblue = "#58849f"
    redish = "#d73027"

    DOT_SIZE = 150

    # create the various dots
    # avg dot
    ax.scatter(
        x=df[f"{metric}_avg"],
        y=df["classifier"],
        s=DOT_SIZE,
        alpha=1,
        label="Average",
        color=lightblue,
        edgecolors="white",
    )

    # min dot
    ax.scatter(
        x=df[f"{metric}_min"],
        y=df["classifier"],
        s=DOT_SIZE,
        alpha=1,
        color=darkblue,
        label="Min/Max",
        edgecolors="white",
        zorder=10,
    )

    # max dot
    ax.scatter(
        x=df[f"{metric}_max"],
        y=df["classifier"],
        s=DOT_SIZE,
        alpha=1,
        color=darkblue,
        edgecolors="white",
    )

    # create the horizontal line
    # between min and max vals
    ax.hlines(
        y=df["classifier"],
        xmin=df[f"{metric}_min"],
        xmax=df[f"{metric}_max"],
        color="grey",
        alpha=0.4,
        lw=4,  # line-width
        zorder=0,  # make sure line at back
    )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # plot the line that shows how a naive classifier performs
    # plot two lines, one white, so that there is a gap between grid lines
    # from https://stackoverflow.com/a/12731750/9214620
    ax.plot([0.024, 0.024], [y_min, y_max], linestyle="-", color="white", linewidth=14)
    ax.plot([0.024, 0.024], [y_min, y_max], linestyle="--", color=redish, alpha=0.4)

    # dictionary used to map the column labels from df to a readable name
    label_dict = {
        "sgd": "SGD Linear",
        "xgb": "XGBoost",
        "rf": "Random Forest",
        "knn": "KNN",
        "nb": "Naive Bayes",
        "ridge": "Ridge Regression",
        "svm": "SVM",
        "lr": "Logistic Regression",
    }

    # iterate through each result and apply the text
    # df should already be sorted
    for i in range(0, df.shape[0]):
        # avg auc score
        ax.text(
            x=df[f"{metric}_avg"][i],
            y=i + 0.15,
            s="{:.2f}".format(df[f"{metric}_avg"][i]),
            horizontalalignment="center",
            verticalalignment="bottom",
            size="x-large",
            color="dimgrey",
            weight="medium",
        )

        # min auc score
        ax.text(
            x=df[f"{metric}_min"][i],
            y=i - 0.15,
            s="{:.2f}".format(df[f"{metric}_min"][i]),
            horizontalalignment="right",
            verticalalignment="top",
            size="x-large",
            color="dimgrey",
            weight="medium",
            backgroundcolor="white",
            zorder=9,
        )

        # max auc score
        ax.text(
            x=df[f"{metric}_max"][i],
            y=i - 0.15,
            s="{:.2f}".format(df[f"{metric}_max"][i]),
            horizontalalignment="left",
            verticalalignment="top",
            size="x-large",
            color="dimgrey",
            weight="medium",
        )

        # add thin leading lines towards classifier names
        # to the right of max dot
        ax.plot(
            [df[f"{metric}_max"][i] + 0.02, 1.0],
            [i, i],
            linewidth=1,
            color="grey",
            alpha=0.4,
            zorder=0,
        )

        # to the left of min dot
        ax.plot(
            [-0.05, df[f"{metric}_min"][i] - 0.02],
            [i, i],
            linewidth=1,
            color="grey",
            alpha=0.4,
            zorder=0,
        )

        # add classifier name text
        clf_name = label_dict[df["classifier"][i]]
        ax.text(
            x=-0.059,
            y=i,
            s=clf_name,
            horizontalalignment="right",
            verticalalignment="center",
            size="x-large",
            color="dimgrey",
            weight="normal",
        )

    # add text for the naive classifier
    ax.text(
        x=0.024,
        y=(y_max - 2),
        s="Naive Classifier",
        horizontalalignment="center",
        verticalalignment="bottom",
        size="large",
        color=redish,
        rotation="vertical",
        backgroundcolor="white",
        alpha=0.4,
        zorder=8,
    )

    # remove the y ticks
    ax.set_yticks([])

    # drop the gridlines (inherited from 'seaborn-whitegrid' style)
    # and drop all the spines
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # custom set the xticks since this looks better
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # set properties of xtick labels
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html#matplotlib.axes.Axes.tick_params
    ax.tick_params(axis="x", pad=20, labelsize="x-large", labelcolor="dimgrey")

    # Add plot title and then description underneath it
    if plt_title is None:
        plt_title = "Top Performing Models by PR-AUC Score"

    plt_desc = (
        "The top performing models in the feature engineering approach, "
        "as sorted by the precision-recall area-under-curve (PR-AUC) score. "
        "The average PR-AUC score for the k-folds-cross-validiation is shown, "
        "along with the minimum and maximum scores in the cross-validation. The baseline"
        " of a naive/random classifier is demonstated by a dotted line."
    )

    # set the plot description
    # use the textwrap.fill (from textwrap std. lib.) to
    # get the text to wrap after a certain number of characters
    PLT_DESC_LOC = 8.8
    # ax.text(
    #     x=-0.05,
    #     y=PLT_DESC_LOC,
    #     s=textwrap.fill(plt_desc, 90),
    #     horizontalalignment="left",
    #     verticalalignment="top",
    #     size="large",
    #     color="dimgrey",
    #     weight="normal",
    #     wrap=True,
    # )

    ax.text(
        x=-0.05,
        y=PLT_DESC_LOC + 0.1,
        s=plt_title,
        horizontalalignment="left",
        verticalalignment="bottom",
        size=16,
        color="dimgrey",
        weight="semibold",
        wrap=True,
    )

    # create legend
    # matplotlib > 3.3.0 can use labelcolor in legend
    # to change color of text
    l = ax.legend(
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        fontsize="x-large",
        # loc="lower center",
        # labelcolor="dimgrey",
    )

    # google colab doesn't use matplotlib > 3.3.0, so we'll change text color
    # the old way, from https://stackoverflow.com/a/18910491/9214620
    for text in l.get_texts():
        text.set_color("dimgrey")

    if save_plot:
        if path_save_dir is None:
            path_save_dir = "./"

        # save as both png and pdf
        plt.savefig(path_save_dir / f"{plt_title}.png", dpi=300, bbox_inches="tight")
        plt.savefig(path_save_dir / f"{plt_title}.pdf", bbox_inches="tight")
        plt.cla()
        plt.close()
    else:
        plt.show()


###############################################################################
# CNC plotting functions
###############################################################################


def plot_features_by_average_index_mpl(
    df,
    feat_to_trend,
    tool_no=54,
    index_list=[1, 2],
    chart_height=9000,
    start_index=1500,
    stop_index=4700,
    path_save_dir=None,
    save_name="feat_trends",
    dpi=300,
    save_plot=False,
):
    """Function to plot the feature table results"""

    # set styling
    sns.set(style="whitegrid", font="DejaVu Sans")

    def convert_to_datetime(cols):
        unix_date = cols[0]
        value = datetime.fromtimestamp(unix_date)
        return value

    df = (
        df[(df["tool_no"] == tool_no) & (df["index_no"].isin(index_list))]
        .groupby(["unix_date"], as_index=False)
        .mean()
    )
    df = df.reset_index(drop=True).sort_values("unix_date")
    df["date"] = df[["unix_date"]].apply(convert_to_datetime, axis=1)
    df["date_ymd"] = pd.to_datetime(df["date"], unit="s").dt.to_period("D")

    df = df[start_index:stop_index]
    df = df.reset_index(drop=True)

    # get date-changes
    # https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy
    v = np.array(df["date_ymd"], dtype=datetime)
    date_change_list = np.where(v[:-1] != v[1:])[0]

    feat_list = []
    feat_title_list = []
    for feat in feat_to_trend:
        feat_list.append([feat, feat_to_trend[feat]])
        feat_title_list.append(feat_to_trend[feat])

    index_str = ""
    index_str_file = ""
    for i in index_list:
        index_str += str(i) + ", "
        index_str_file += str(i) + "_"

    title_chart = (
        "Features for Tool {}, Averaged Across Splits (splits on metal-cutting)".format(
            tool_no
        )
    )
    file_name = "tool_{}_avg_splits_{}.pdf".format(tool_no, index_str_file[:-1])

    cols = 1
    rows = int(len(feat_to_trend) / cols)
    if (len(feat_to_trend) % cols) != 0:
        rows += 1

    fig, ax = plt.subplots(rows, cols, figsize=(5, 3.5), dpi=600)
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)

    # get lenght of features
    l = len(feat_to_trend)
    len_data = len(df)

    # iterate through each feature number and plot on chart
    counter = 0
    for i in range(rows):
        for j in range(cols):
            if counter < l:
                trend_value = feat_list[counter][0]
                title_plot = feat_title_list[counter]

                min_plot_val = np.min(df[trend_value])
                max_plot_val = np.max(df[trend_value])
                len_trend_data = len(df[trend_value])

                if title_plot != feat_title_list[counter]:

                    print("ERROR IN SUB-PLOT TITLE")

                for failed_label in range(0, 2):
                    df1 = df[df["y"] == failed_label].copy()

                    color_label = [pal[3], "#e31a1c", "#b2df8a"]
                    color_outline = [pal[5], "#b51516", "#b51516"]

                    x = df1.index.values
                    y = df1[trend_value]

                    p = df1[["unix_date", "date"]].values
                    point_name = []
                    for d in p:
                        if failed_label == 1:
                            point_name.append(
                                "Label: Wear, " + str(d[1]) + ", " + str(d[0])
                            )
                        elif failed_label == 2:
                            point_name.append(
                                "Label: Ignore, " + str(d[1]) + ", " + str(d[0])
                            )
                        else:
                            point_name.append(str(d[1]) + ", " + str(d[0]))

                    ax[i].scatter(
                        x,
                        y,
                        s=4,
                        color=color_label[failed_label],
                        alpha=0.4,
                        linewidths=0,
                        edgecolors=color_outline[failed_label],
                        label=failed_label,
                    )

                # add the vertical lines
                for date_change in date_change_list:
                    date_change_text = str(df["date_ymd"].to_numpy()[date_change + 1])
                    ax[i].axvline(
                        date_change,
                        ymin=0,
                        ymax=6000,
                        color="k",
                        alpha=0.3,
                        linestyle="--",
                        zorder=0,
                        linewidth=0.75,
                    )

                    # DATE LABELS
                    if counter == (l - 1):
                        ax[i].text(
                            date_change + 15,
                            min_plot_val - (max_plot_val - min_plot_val) * 0.09,
                            date_change_text,
                            rotation=45,
                            size=4,
                            va="top",
                            ha="right",
                        )

                    axis_label = feat_title_list[i]
                    ax[i].set_ylabel(
                        axis_label,
                        fontsize=5,
                        ma="right",
                        horizontalalignment="right",
                        verticalalignment="center",
                    ).set_rotation(0)

                ax[i].set_xlim(df.index[0] - 25, df.index[-1] + 25)
                counter += 1
            else:
                pass

    for axes in ax.flatten():
        spine_width = 0.5
        axes.xaxis.set_tick_params(labelbottom=False)
        axes.yaxis.set_tick_params(labelleft=False, which="major")
        axes.grid(False)
        axes.spines["top"].set_linewidth(spine_width)
        axes.spines["bottom"].set_linewidth(spine_width)
        axes.spines["left"].set_linewidth(spine_width)
        axes.spines["right"].set_linewidth(spine_width)
    #         axes.axis('off')

    L = plt.legend(
        # bbox_to_anchor=(0, -0.7),
        loc="lower right",
        ncol=2,
        frameon=True,
        fontsize=4,
    )
    L.get_texts()[0].set_text("Healthy")
    L.get_texts()[1].set_text("Failed")
    L.get_frame().set_linewidth(0.1)
    # L.get_texts()[2].set_text('_Hidden')

    if save_plot:
        # matplotlib.use('Agg') # use this if on HPC
        if path_save_dir is None:
            path_save_dir = Path.cwd()

        plt.savefig(
            path_save_dir / f"{save_name}.pdf",
            bbox_inches="tight",
        )

        plt.savefig(path_save_dir / f"{save_name}.png", bbox_inches="tight", dpi=dpi)
        plt.cla()
        plt.close()
    else:
        plt.show()


###############################################################################
# Main functions to plot datasets
###############################################################################


def set_directories(args):
    """Sets the directories for the raw, interim, and processed data."""

    assert (
        args.dataset == "cnc" or args.dataset == "milling"
    ), "Dataset must be either 'cnc' or 'milling'"

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    path_save_dir = proj_dir / "reports" / "figures"
    path_save_dir.mkdir(parents=True, exist_ok=True)

    return (
        proj_dir,
        path_data_dir,
        path_save_dir,
    )


###############################################################################
#### CNC DATASET
###############################################################################
def plot_cnc_data(
    proj_dir,
    path_data_dir,
    path_save_dir,
    processed_dir_name,
    feat_file_name,
    save_plot=False,
):

    path_processed_dir = path_data_dir / "processed" / "cnc" / processed_dir_name

    # load feature df and df_labels and merge
    df = pd.read_csv(path_processed_dir / feat_file_name)
    df["unix_date"] = df["id"].apply(lambda x: int(x.split("_")[0]))
    df["tool_no"] = df["id"].apply(lambda x: int(x.split("_")[-2]))
    df["index_no"] = df["id"].apply(lambda x: int(x.split("_")[-1]))

    df_labels = pd.read_csv(
        path_data_dir
        / "processed"
        / "cnc"
        / "high_level_labels_MASTER_update2020-08-06_new-jan-may-data_with_case.csv"
    )

    df = cnc_add_y_label_binary(df, df_labels, col_list_case=["case_tool_54"])
    df = df.dropna(axis=0)

    ###################
    # Trend features

    feat_to_trend = {
        'current__fft_coefficient__attr_"abs"__coeff_58': "fft coeff. 58",
        'current__fft_coefficient__attr_"abs"__coeff_97': "fft coeff. 97",
        'current__fft_coefficient__attr_"imag"__coeff_45': "fft coeff. 45",
        'current__fft_coefficient__attr_"real"__coeff_12': "fft coeff. 12",
        'current__fft_coefficient__attr_"imag"__coeff_77': "fft coeff. 77",
        'current__fft_coefficient__attr_"abs"__coeff_49': "fft coeff. 49",
    }

    plot_features_by_average_index_mpl(
        df,
        feat_to_trend=feat_to_trend,
        tool_no=54,
        index_list=[2, 3, 4, 5, 6, 7],
        chart_height=9000,
        start_index=1000,
        stop_index=4900,
        path_save_dir=path_save_dir,
        save_name="feat_trends",
        dpi=300,
        save_plot=save_plot,
    )

    ###################
    # Lollipop plot

    df_results = pd.read_csv(
        proj_dir
        / "models/final_results_cnc_2022_08_04_final"
        / "compiled_results_filtered_best.csv"
    )

    plot_lollipop_results(
        df_results,
        metric="prauc",
        plt_title=None,
        path_save_dir=path_save_dir,
        save_name="results_lollipop",
        save_plot=True,
        dpi=300,
    )


def main(args):

    (
        proj_dir,
        path_data_dir,
        path_save_dir,
    ) = set_directories(args)

    plot_cnc_data(
        proj_dir,
        path_data_dir,
        path_save_dir,
        processed_dir_name="cnc_features_custom_1",
        feat_file_name="cnc_features_54_custom_1.csv",
        save_plot=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build data sets for analysis")

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--path_data_dir",
        dest="path_data_dir",
        type=str,
        help="Location of the data folder, containing the raw, interim, and processed folders",
    )

    parser.add_argument(
        "--processed_dir_name",
        default="features",
        type=str,
        help="Name of the save directory. Used to store features. Located in data/processed/cnc",
    )

    parser.add_argument(
        "--dataset",
        default="milling",
        type=str,
        help="Name of the dataset to use for training. Either 'milling' or 'cnc'",
    )

    args = parser.parse_args()

    main(args)
