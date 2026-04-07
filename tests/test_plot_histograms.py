from geci_plots.plot_histograms import plot_histogram_with_limits
import matplotlib as plt
import pandas as pd


def test_plot_histogram_with_limits():
    data = pd.read_csv("tests/data/monthly_data.csv")
    obtained = plot_histogram_with_limits(data["Effort"], None)
    assert isinstance(obtained, plt.axes._axes.Axes)
