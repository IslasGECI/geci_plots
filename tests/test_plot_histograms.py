from geci_plots.plot_histograms import plot_histogram_with_limits
import matplotlib as plt
import pandas as pd


def test_plot_histogram_with_limits():
    data = pd.read_csv("tests/data/monthly_data.csv")
    column_name = "Effort"
    x = data[column_name]

    obtained = plot_histogram_with_limits(x, None)
    assert isinstance(obtained, plt.axes._axes.Axes)
    assert obtained.get_xaxis().get_label().get_fontsize() == 20

    expected_xlabel_fontsize = 16
    plot_options = {"label": column_name, "fontsize": expected_xlabel_fontsize}
    limits = [47698]
    color_line = "r"
    lines_options = {"color": color_line}
    obtained = plot_histogram_with_limits(
        x, None, limits=limits, plot_options=plot_options, lines_options=lines_options
    )
    assert isinstance(obtained, plt.axes._axes.Axes)

    assert obtained.get_xlabel() == column_name
    assert obtained.get_lines()[0].get_data()[0][0] == limits[0]
    assert obtained.get_lines()[0].get_color() == color_line
    assert obtained.get_xaxis().get_label().get_fontsize() == expected_xlabel_fontsize

    expected_fontsize = 15
    assert obtained.get_xticklabels()[0].get_fontsize() == expected_fontsize
    assert obtained.get_yticklabels()[0].get_fontsize() == expected_fontsize
