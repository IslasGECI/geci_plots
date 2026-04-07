from geci_plots.plot_histograms import plot_histogram_with_limits
import matplotlib as plt
import pandas as pd


def test_plot_histogram_with_limits():
    data = pd.read_csv("tests/data/monthly_data.csv")
    column_name = "Effort"
    x = data[column_name]
    plot_options = {"label": column_name}
    limits = [47698]
    color_line = "r"
    lines_options = {"color": color_line}
    obtained = plot_histogram_with_limits(
        x, None, limits=limits, plot_options=plot_options, lines_options=lines_options
    )
    assert isinstance(obtained, plt.axes._axes.Axes)

    assert obtained.get_xlabel() == column_name
    plt.pyplot.savefig("histogram.png")
    assert obtained.get_lines()[0].get_data()[0][0] == limits[0]
    assert obtained.get_lines()[0].get_color() == color_line
