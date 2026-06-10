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
    plt.pyplot.savefig("radar_signal.png")
    assert isinstance(obtained, plt.axes._axes.Axes)
    assert len(obtained.collections) == 3
    scatter_collection = obtained.collections[2]
    expected_colormap_limits = (
        radar_signal_points_data["radar_signal"].min(),
        radar_signal_points_data["radar_signal"].max(),
    )
    assert scatter_collection.get_clim() == expected_colormap_limits
    assert scatter_collection.get_cmap().name == "cool"
    assert scatter_collection.get_alpha() == 0.9
    assert scatter_collection.get_sizes().min() == expected_colormap_limits[0] * 0.5
    assert isinstance(scatter_collection.norm, plt.colors.LogNorm)

    colorbar_object = obtained.get_figure().get_children()[2]
    colorbar_object_label = colorbar_object.get_ylabel()
    expected_colorbal_label = "Radar Signal Strength"
    assert colorbar_object_label == expected_colorbal_label
    expected_colorbar_fontsize = 20
    assert colorbar_object.get_yaxis().label.get_fontsize() == expected_colorbal_label
    assert colorbar_object._colorbar_info["location"] == "left"
    assert colorbar_object.get_yticklabels()[0].get_fontsize() == expected_colorbar_fontsize
