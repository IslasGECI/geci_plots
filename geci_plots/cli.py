import pandas as pd
import typer
import matplotlib.pyplot as plt

cli = typer.Typer()


@cli.command()
def plot_kernel_density_gls():
    pass


@cli.command()
def version():
    print("0.4.1")
