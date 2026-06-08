from geci_plots.plot_kernel_density_gls import (
    _plot_geographic_points,
    _plot_geographic_points_by_trip,
    _plot_geographic_points_by_vessel,
    _plot_kernel_density,
    _plot_kernel_density_and_points,
    adapt_geographic_data,
)
from geci_plots.boxplots import create_box_plot, create_box_plot_data_from_columns
from geci_plots.process_cameras_data import _plot_monthly_cameras_effort_and_captures
from geci_plots.process_effort_and_captures_data import (
    _plot_monthly_traps_effort_and_captures_by_zone,
)
from geci_plots.plot_population_time_series import plot_population_time_series
from geci_plots.plot_histograms import plot_histogram_with_limits
from geci_plots.plot_radar_signal import plot_radar_signal_geographic_points

import geci_plots as gp
import matplotlib.pyplot as plt
import pandas as pd
import typer
import json

cli = typer.Typer()


@cli.command()
def render_radar_signal_geographic_points(
    radar_signal_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    result_map_path: str = typer.Option(),
):
    geographic_data = pd.read_csv(radar_signal_data_path)
    plot_radar_signal_geographic_points(geographic_data, global_shapefile_data_path, path_rose_wind)
    plt.savefig(result_map_path, transparent=True)


@cli.command()
def render_population_time_series(
    data_path: str = typer.Option(), output_path: str = typer.Option()
):
    data_dictionary = read_json(data_path)
    data = pd.DataFrame(data_dictionary["time_series"])
    plot_population_time_series(data)
    plt.savefig(output_path, transparent=True)


@cli.command()
def render_histogram_with_median(
    data_path: str = typer.Option(),
    column_name: str = typer.Option(),
    output_path: str = typer.Option(),
):
    data_df = pd.read_csv(data_path)
    x_values = data_df[column_name]
    limits = [x_values.median()]
    plot_options = {"label": column_name}
    lines_options = {"color": "r"}
    plot_histogram_with_limits(
        x_values, None, limits=limits, plot_options=plot_options, lines_options=lines_options
    )
    plt.savefig(output_path, transparent=True)


def read_json(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)
    return data


@cli.command()
def plot_monthly_traps_effort_and_captures_by_zone(
    effort_captures_data_path: str = typer.Option(),
    start_date: str = typer.Option(),
    end_date: str = typer.Option(),
    output_path: str = typer.Option(),
):
    effort_captures_data = pd.read_csv(effort_captures_data_path)
    _plot_monthly_traps_effort_and_captures_by_zone(effort_captures_data, start_date, end_date)
    plt.savefig(output_path, transparent=True)


@cli.command()
def plot_monthly_cameras_effort_and_captures(
    cameras_data_path: str = typer.Option(),
    start_date: str = typer.Option(),
    end_date: str = typer.Option(),
    output_path: str = typer.Option(),
):
    cameras_data = pd.read_csv(cameras_data_path)
    _plot_monthly_cameras_effort_and_captures(cameras_data, start_date, end_date)
    plt.savefig(output_path, transparent=True)


@cli.command()
def plot_kernel_density_and_points(
    geographic_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    selected_contour: str = typer.Option(),
    result_map_path: str = typer.Option(),
    bandwidth: float = typer.Option(),
):
    geographic_data = adapt_geographic_data(geographic_data_path)
    _plot_kernel_density_and_points(
        geographic_data, global_shapefile_data_path, path_rose_wind, selected_contour, bandwidth
    )
    plt.savefig(result_map_path, transparent=True)


@cli.command()
def plot_kernel_density(
    geographic_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    selected_contour: str = typer.Option(),
    result_map_path: str = typer.Option(),
    bandwidth: float = typer.Option(),
):
    geographic_data = adapt_geographic_data(geographic_data_path)
    _plot_kernel_density(
        geographic_data, global_shapefile_data_path, path_rose_wind, selected_contour, bandwidth
    )
    plt.savefig(result_map_path, transparent=True)


@cli.command()
def plot_geographic_points(
    geographic_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    result_map_path: str = typer.Option(),
):
    geographic_data = adapt_geographic_data(geographic_data_path)
    _plot_geographic_points(geographic_data, global_shapefile_data_path, path_rose_wind)
    plt.savefig(result_map_path, transparent=True)


@cli.command()
def plot_geographic_points_by_trip(
    geographic_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    result_map_path: str = typer.Option(),
):
    geographic_data = adapt_geographic_data(geographic_data_path)
    _plot_geographic_points_by_trip(geographic_data, global_shapefile_data_path, path_rose_wind)
    plt.savefig(result_map_path, transparent=True)


@cli.command()
def plot_geographic_points_by_vessel(
    geographic_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    result_map_path: str = typer.Option(),
):
    geographic_data = adapt_geographic_data(geographic_data_path)
    _plot_geographic_points_by_vessel(geographic_data, global_shapefile_data_path, path_rose_wind)
    plt.savefig(result_map_path, transparent=True)


@cli.command()
def boxplot(
    data_path: str = typer.Option(),
    columns_of_interest: str = typer.Option(),
    boxplot_path: str = typer.Option(),
):
    summary_data = pd.read_csv(data_path)
    columns = columns_of_interest.split(",")
    print(columns)
    data_for_boxplot = create_box_plot_data_from_columns(summary_data, columns)
    create_box_plot(data_for_boxplot)
    plt.savefig(boxplot_path, transparent=True)


@cli.command()
def version():
    print(gp.__version__)
