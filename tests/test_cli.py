from geci_plots import cli

import geci_test_tools as gtt
from typer.testing import CliRunner


runner = CliRunner()


def test_plot_kernel_density_gls():
    result = runner.invoke(cli, ["plot-kernel-density-gls", "--help"])
    assert result.exit_code == 0

    gls_data_path = "tests/data/gls_albatros_tests.csv"
    global_shapefile_data_path = "tests/data/division_politica_paises.shp"
    path_rose_wind = "tests/data/rosewind.png"
    selected_contour = "50_contour"
    result_map_path = "tests/kernel_50_percent_gls_albatros.png"

    gtt.if_exist_remove(result_map_path)
    result = runner.invoke(
        cli,
        [
            "plot-kernel-density-gls",
            "--gls-data-path",
            gls_data_path,
            "--global-shapefile-data-path",
            global_shapefile_data_path,
            "--path-rose-wind",
            path_rose_wind,
            "--selected-contour",
            selected_contour,
            "--result-map-path",
            result_map_path,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(result_map_path)

    gls_data_path = "tests/data/trips_geographic_points_tests.csv"
    result_map_path = "tests/kernel_50_percent_gps_albatros.png"
    gtt.if_exist_remove(result_map_path)
    result = runner.invoke(
        cli,
        [
            "plot-kernel-density-gls",
            "--gls-data-path",
            gls_data_path,
            "--global-shapefile-data-path",
            global_shapefile_data_path,
            "--path-rose-wind",
            path_rose_wind,
            "--selected-contour",
            selected_contour,
            "--result-map-path",
            result_map_path,
        ],
    )
    assert result.exit_code == 0
    gtt.assert_exist(result_map_path)


def test_version():
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    expected_version = "0.4.1"
    assert expected_version in result.stdout
