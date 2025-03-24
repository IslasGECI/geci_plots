from geci_plots.plot_kernel_density_gls import (
    adapt_gls_data,
    _plot_kernel_density_gls,
)

import matplotlib as plt
import pandas as pd


def test_plot_kernel_density_gls():
    gls_data_path = "tests/data/gls_albatros_tests.csv"
    global_shapefile_data_path = "tests/data/division_politica_paises.shp"
    path_rose_wind = "tests/data/rosewind.png"
    selected_contour = "50_contour"

    gls_data = pd.read_csv(gls_data_path)
    obtained = _plot_kernel_density_gls(
        gls_data, global_shapefile_data_path, path_rose_wind, selected_contour
    )
    assert isinstance(obtained, plt.axes._axes.Axes)


def test_plot_kernel_density():
    gls_data_path = "tests/data/trips_geographic_points_tests.csv"
    global_shapefile_data_path = "tests/data/division_politica_paises.shp"
    path_rose_wind = "tests/data/rosewind.png"
    selected_contour = "50_contour"

    gls_data = adapt_gls_data(gls_data_path)
    gls_data = gls_data.sample(frac=0.1)
    obtained = _plot_kernel_density_gls(
        gls_data, global_shapefile_data_path, path_rose_wind, selected_contour
    )
    assert isinstance(obtained, plt.axes._axes.Axes)
