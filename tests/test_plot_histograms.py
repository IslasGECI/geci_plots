from geci_plots.plot_histograms import plot_histogram_with_limits
import matplotlib as plt
import pandas as pd


def test_plot_histogram_with_limits():
    data = pd.read_csv("tests/data/monthly_data.csv")
    column_name = "Effort"
    obtained = plot_histogram_with_limits(data[column_name], None)
    assert isinstance(obtained, plt.axes._axes.Axes)

    assert obtained.get_xlabel() == column_name
