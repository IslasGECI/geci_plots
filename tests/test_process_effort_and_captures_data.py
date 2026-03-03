from geci_plots.process_effort_and_captures_data import (
    process_effort_and_captures_data_by_zone,
    _plot_monthly_traps_effort_and_captures_by_zone,
)
import pandas as pd
import matplotlib as plt

effort_captures_data = pd.read_csv(
    "tests/data/monthly_effort_and_captures_by_zone.csv")
starting_date = "2026-01-01"
ending_date = "2026-03-31"


def test_process_effort_and_captures_data_by_zone():
    obtained = process_effort_and_captures_data_by_zone(
        effort_captures_data, starting_date, ending_date
    )
    expected_number_of_months = 3
    expected_number_of_column_zones = 3
    assert obtained.shape == (
        expected_number_of_months, expected_number_of_column_zones)


def test_plot_monthly_traps_effort_and_captures_by_zone():
    obtained_ax, obtained_ax2 = _plot_monthly_traps_effort_and_captures_by_zone(
        effort_captures_data, starting_date, ending_date
    )
    assert isinstance(obtained_ax, plt.axes._axes.Axes)

    expected_ylabel = "Captures per month (No. cats dispatched)"
    assert obtained_ax.get_ylabel() == expected_ylabel

    expected_ylabel2 = "Effort per month (night traps)"
    assert obtained_ax2.get_ylabel() == expected_ylabel2
    obtained_ax.savefig("plot_monthly_traps_effort_and_captures_by_zone.png")
