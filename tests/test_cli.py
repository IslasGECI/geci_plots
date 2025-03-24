from geci_plots import cli

import geci_test_tools as gtt
from typer.testing import CliRunner


runner = CliRunner()


def test_plot_kernel_density_gls():
    result = runner.invoke(cli, ["plot-kernel-density-gls", "--help"])
    assert result.exit_code == 0
