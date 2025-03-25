from geci_plots.plot_kernel_density_gls import (
    adapt_gls_data,
    _plot_kernel_density,
    _plot_kernel_density_and_points,
    _plot_geographic_points
)

import matplotlib as plt
import pandas as pd


global_shapefile_data_path = "tests/data/division_politica_paises.shp"
path_rose_wind = "tests/data/rosewind.png"
selected_contour = "50_contour"
bandwidth = 0.04
gls_data_path = "tests/data/gls_albatros_tests.csv"
gps_data_path = "tests/data/trips_geographic_points_tests.csv"


def test_plot_kernel_density_and_gls_points():
    gls_data = pd.read_csv(gls_data_path)
    obtained = _plot_kernel_density_and_points(
        gls_data, global_shapefile_data_path, path_rose_wind, selected_contour, bandwidth
    )
    assert isinstance(obtained, plt.axes._axes.Axes)


def test_plot_kernel_density_and_points():
    gps_data = adapt_gls_data(gps_data_path)
    obtained = _plot_kernel_density_and_points(
        gps_data, global_shapefile_data_path, path_rose_wind, selected_contour, bandwidth
    )
    assert isinstance(obtained, plt.axes._axes.Axes)
    assert len(obtained.collections) == 3
    assert len(obtained.lines) == 1


def test_plot_kernel_density():
    gps_data = adapt_gls_data(gps_data_path)
    obtained = _plot_kernel_density(
        gps_data, global_shapefile_data_path, path_rose_wind, selected_contour, bandwidth
    )
    plt.pyplot.savefig("kernel.png")
    assert isinstance(obtained, plt.axes._axes.Axes)
    assert len(obtained.collections) == 3
    assert len(obtained.lines) == 0


def test_plot_geographic_points():
    gls_data = pd.read_csv(gls_data_path)
    obtained = _plot_geographic_points(gls_data, global_shapefile_data_path, path_rose_wind)
    plt.pyplot.savefig("points.png")
    assert isinstance(obtained, plt.axes._axes.Axes)
    assert len(obtained.collections) == 3
    assert len(obtained.lines) == 1
