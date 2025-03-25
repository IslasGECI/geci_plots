from geci_plots.plot_kernel_density_gls import (
    adapt_gls_data,
    _plot_geographic_points,
    _plot_kernel_density_and_points,
    _plot_kernel_density,
)
import typer
import matplotlib.pyplot as plt

cli = typer.Typer()


@cli.command()
def plot_kernel_density_and_points(
    gls_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    selected_contour: str = typer.Option(),
    result_map_path: str = typer.Option(),
    bandwidth: float = typer.Option(),
):
    geographic_data = adapt_gls_data(gls_data_path)
    _plot_kernel_density_and_points(
        geographic_data, global_shapefile_data_path, path_rose_wind, selected_contour, bandwidth
    )
    plt.savefig(result_map_path)


@cli.command()
def plot_kernel_density(
    gls_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    selected_contour: str = typer.Option(),
    result_map_path: str = typer.Option(),
    bandwidth: float = typer.Option(),
):
    geographic_data = adapt_gls_data(gls_data_path)
    _plot_kernel_density(
        geographic_data, global_shapefile_data_path, path_rose_wind, selected_contour, bandwidth
    )
    plt.savefig(result_map_path)


@cli.command()
def plot_geographic_points(
    gls_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    result_map_path: str = typer.Option(),
):
    geographic_data = adapt_gls_data(gls_data_path)
    _plot_geographic_points(geographic_data, global_shapefile_data_path, path_rose_wind)
    plt.savefig(result_map_path)


@cli.command()
def version():
    print("0.4.1")
