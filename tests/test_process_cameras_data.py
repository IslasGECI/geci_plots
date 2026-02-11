from geci_plots.process_cameras_data import summarize_monthly_cameras_effort_and_captures

import pandas as pd


def test_summarize_monthly_cameras_effort_and_captures():
    camera_data = pd.read_csv("tests/data/weekly_cameras_effort.csv")
    begin_date = "2025-01-01"
    end_date = "2026-01-04"
    obtained = summarize_monthly_cameras_effort_and_captures(camera_data, begin_date, end_date)
    expected_nrows = 4
    assert len(obtained) == expected_nrows
    assert obtained.loc["2026-01-01", "Effort"] == 182
