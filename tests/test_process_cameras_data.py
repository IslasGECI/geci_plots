from geci_plots.process_cameras_data import (
    summarize_monthly_cameras_effort_and_captures,
    _plot_monthly_cameras_effort_and_captures,
)

import pandas as pd
import matplotlib as plt

camera_data = pd.read_csv("tests/data/weekly_cameras_effort.csv")
begin_date = "2025-01-01"
end_date = "2026-01-04"


def test_plot_monthly_cameras_effort_and_captures():
    obtained_ax, obtained_ax2 = _plot_monthly_cameras_effort_and_captures(
        camera_data, begin_date, end_date
    )
    assert isinstance(obtained_ax, plt.axes._axes.Axes)

    expected_ylabel = "Detections per month (cats in photos)"
    assert obtained_ax.get_ylabel() == expected_ylabel

    expected_ylabel2 = "Effort per month (camera night traps)"
    assert obtained_ax2.get_ylabel() == expected_ylabel2

    assert obtained_ax2.get_shared_x_axes().joined(obtained_ax, obtained_ax2)
    assert obtained_ax.bbox.bounds == obtained_ax2.bbox.bounds


def test_summarize_monthly_cameras_effort_and_captures():
    obtained = summarize_monthly_cameras_effort_and_captures(camera_data, begin_date, end_date)
    expected_nrows = 4
    assert len(obtained) == expected_nrows
    assert obtained.loc["2026-01-01", "Effort"] == 182
