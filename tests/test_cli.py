from geci_plots import cli

import geci_test_tools as gtt
from typer.testing import CliRunner


runner = CliRunner()


gls_data_path = "tests/data/gls_albatros_tests.csv"
gps_data_path = "tests/data/trips_geographic_points_tests.csv"
global_shapefile_data_path = "tests/data/division_politica_paises.shp"
path_rose_wind = "tests/data/rosewind.png"
bandwidth = 0.04


def test_plot_kernel_density_gls():
    result = runner.invoke(cli, ["plot-kernel-density-and-points", "--help"])
    assert result.exit_code == 0

    selected_contour = "50_contour"
    result_map_path = "tests/kernel_50_percent_gls_albatros.png"

    gtt.if_exist_remove(result_map_path)
    result = runner.invoke(
        cli,
        [
            "plot-kernel-density-and-points",
            "--geographic-data-path",
            gls_data_path,
            "--global-shapefile-data-path",
            global_shapefile_data_path,
            "--path-rose-wind",
            path_rose_wind,
            "--selected-contour",
            selected_contour,
            "--result-map-path",
            result_map_path,
            "--bandwidth",
            bandwidth,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(result_map_path)

    result_map_path = "tests/kernel_50_percent_gps_albatros.png"
    gtt.if_exist_remove(result_map_path)
    result = runner.invoke(
        cli,
        [
            "plot-kernel-density-and-points",
            "--geographic-data-path",
            gps_data_path,
            "--global-shapefile-data-path",
            global_shapefile_data_path,
            "--path-rose-wind",
            path_rose_wind,
            "--selected-contour",
            selected_contour,
            "--result-map-path",
            result_map_path,
            "--bandwidth",
            bandwidth,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(result_map_path)


def test_plot_kernel_density():
    result = runner.invoke(cli, ["plot-kernel-density", "--help"])
    assert result.exit_code == 0

    gls_data_path = "tests/data/gls_albatros_tests.csv"
    global_shapefile_data_path = "tests/data/division_politica_paises.shp"
    path_rose_wind = "tests/data/rosewind.png"
    selected_contour = "All_contours"
    result_map_path = "tests/kernel_100_percent_gls_albatros.png"
    bandwidth = 0.1

    gtt.if_exist_remove(result_map_path)
    result = runner.invoke(
        cli,
        [
            "plot-kernel-density",
            "--geographic-data-path",
            gls_data_path,
            "--global-shapefile-data-path",
            global_shapefile_data_path,
            "--path-rose-wind",
            path_rose_wind,
            "--selected-contour",
            selected_contour,
            "--result-map-path",
            result_map_path,
            "--bandwidth",
            bandwidth,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(result_map_path)


def test_plot_geographic_points():
    result = runner.invoke(cli, ["plot-geographic-points", "--help"])
    assert result.exit_code == 0

    result_map_path = "tests/geographic_points_gls_albatros.png"
    gtt.if_exist_remove(result_map_path)
    result = runner.invoke(
        cli,
        [
            "plot-geographic-points",
            "--geographic-data-path",
            gls_data_path,
            "--global-shapefile-data-path",
            global_shapefile_data_path,
            "--path-rose-wind",
            path_rose_wind,
            "--result-map-path",
            result_map_path,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(result_map_path)


def test_plot_geographic_points_by_trip():
    result = runner.invoke(cli, ["plot-geographic-points-by-trip", "--help"])
    assert result.exit_code == 0

    result_map_path = "tests/geographic_points_gps_albatros_by_trip.png"
    gtt.if_exist_remove(result_map_path)
    result = runner.invoke(
        cli,
        [
            "plot-geographic-points-by-trip",
            "--geographic-data-path",
            gps_data_path,
            "--global-shapefile-data-path",
            global_shapefile_data_path,
            "--path-rose-wind",
            path_rose_wind,
            "--result-map-path",
            result_map_path,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(result_map_path)


def test_plot_geographic_points_by_vessel():
    result = runner.invoke(cli, ["plot-geographic-points-by-vessel", "--help"])
    assert result.exit_code == 0

    vessels_gps_data_path = "tests/data/vessels_geographic_points_tests.csv"
    result_map_path = "tests/geographic_points_gps_by_vessel.png"
    gtt.if_exist_remove(result_map_path)
    result = runner.invoke(
        cli,
        [
            "plot-geographic-points-by-vessel",
            "--geographic-data-path",
            vessels_gps_data_path,
            "--global-shapefile-data-path",
            global_shapefile_data_path,
            "--path-rose-wind",
            path_rose_wind,
            "--result-map-path",
            result_map_path,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(result_map_path)


def test_boxplot():
    result = runner.invoke(cli, ["boxplot", "--help"])
    assert result.exit_code == 0

    data_path = "tests/data/summary_trips.csv"
    boxplot_path = "tests/boxplot.png"
    columns_of_interest = "duration,total_dist,max_dist"
    gtt.if_exist_remove(boxplot_path)
    result = runner.invoke(
        cli,
        [
            "boxplot",
            "--data-path",
            data_path,
            "--columns-of-interest",
            columns_of_interest,
            "--boxplot-path",
            boxplot_path,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(boxplot_path)


def test_version():
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    expected_version = "0.7.0"
    assert expected_version in result.stdout
