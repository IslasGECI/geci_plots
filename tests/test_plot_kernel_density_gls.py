from geci_plots.plot_kernel_density_gls import (
    adapt_geographic_data,
    format_plot,
    _plot_kernel_density,
    _plot_kernel_density_and_points,
    _plot_geographic_points,
    _plot_geographic_points_by_trip,
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
    gps_data = adapt_geographic_data(gps_data_path)
    obtained = _plot_kernel_density_and_points(
        gps_data, global_shapefile_data_path, path_rose_wind, selected_contour, bandwidth
    )
    assert isinstance(obtained, plt.axes._axes.Axes)
    assert len(obtained.collections) == 3
    assert len(obtained.lines) == 1


def test_plot_kernel_density():
    gps_data = adapt_geographic_data(gps_data_path)
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
    assert len(obtained.collections) == 2
    assert len(obtained.lines) == 1


def test_plot_geographic_points_by_trip():
    gps_data_path = "trips_geographic_points.csv"
    gps_data = adapt_geographic_data(gps_data_path)
    obtained = _plot_geographic_points_by_trip(gps_data, global_shapefile_data_path, path_rose_wind)
    plt.pyplot.savefig("points_by_trip.png")
    assert isinstance(obtained, plt.axes._axes.Axes)
    assert len(obtained.collections) == 2
    assert len(obtained.lines) > 1
    assert obtained.lines[0].get_marker() == "."
    assert obtained.lines[0].get_markersize() == 3


def test_format_plot():
    fig, ax = plt.pyplot.subplots(figsize=(14.3, 10.4))
    gps_data = adapt_geographic_data(gps_data_path)
    format_plot(ax, gps_data)
    assert ax.get_ylim()[1] == 60
    assert ax.get_ylim()[0] == 20
    assert ax.get_xlim()[1] == -110
    assert ax.get_xlim()[0] == -160
