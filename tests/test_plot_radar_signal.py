from geci_plots.plot_radar_signal import plot_radar_signal_geographic_points

import matplotlib as plt
import pandas as pd


def test_plot_radar_signal_geographic_points():
    radar_signal_points_data = pd.read_csv("tests/data/radar_signal_with_coordinates.csv")
    global_shapefile_data_path = "tests/data/division_politica_paises.shp"
    path_rose_wind = "tests/data/rosewind.png"
    obtained = plot_radar_signal_geographic_points(
        radar_signal_points_data, global_shapefile_data_path, path_rose_wind
    )
    assert isinstance(obtained, plt.axes._axes.Axes)
