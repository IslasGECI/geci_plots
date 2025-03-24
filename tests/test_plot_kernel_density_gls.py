import geci_test_tools as gtt

from geci_plots.plot_kernel_density_gls import (
    adapt_gls_data,
    _plot_kernel_density_gls,
    X_plot_kernel_density_gls,
)


def test_plot_kernel_density_gls():
    gls_data_path = "tests/data/gls_albatros_tests.csv"
    global_shapefile_data_path = "tests/data/division_politica_paises.shp"
    path_rose_wind = "tests/data/rosewind.png"
    selected_contour = "50_contour"
    result_map_path = "tests/kernel_50_percent_gls_albatros.png"

    gtt.if_exist_remove(result_map_path)
    gls_data = adapt_gls_data(gls_data_path)
    X_plot_kernel_density_gls(
        gls_data, global_shapefile_data_path, path_rose_wind, selected_contour, result_map_path
    )
    gtt.assert_exist(result_map_path)


def test_plot_kernel_density():
    gls_data_path = "tests/data/trips_geographic_points_tests.csv"
    global_shapefile_data_path = "tests/data/division_politica_paises.shp"
    path_rose_wind = "tests/data/rosewind.png"
    selected_contour = "50_contour"
    result_map_path = "tests/kernel_50_percent_gps_albatros.png"

    gtt.if_exist_remove(result_map_path)
    _plot_kernel_density_gls(
        gls_data_path, global_shapefile_data_path, path_rose_wind, selected_contour, result_map_path
    )
    gtt.assert_exist(result_map_path)
