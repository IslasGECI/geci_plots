from geci_plots.plot_kernel_density_gls import adapt_gls_data, _plot_kernel_density_gls
import pandas as pd
import typer
import matplotlib.pyplot as plt

cli = typer.Typer()


@cli.command()
def plot_kernel_density_gls(
    gls_data_path: str = typer.Option(),
    global_shapefile_data_path: str = typer.Option(),
    path_rose_wind: str = typer.Option(),
    selected_contour: str = typer.Option(),
    result_map_path: str = typer.Option(),
):
    gls_data = adapt_gls_data(gls_data_path)
    _plot_kernel_density_gls(
        gls_data, global_shapefile_data_path, path_rose_wind, selected_contour, result_map_path
    )


@cli.command()
def version():
    print("0.4.1")
