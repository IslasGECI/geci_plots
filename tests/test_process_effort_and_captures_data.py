from geci_plots.process_effort_and_capures_data import process_effort_and_captures_data_by_zone
import pandas as pd


def test_process_effort_and_captures_data_by_zone():
    effort_captures_data = pd.read_csv("tests/data/monthly_effort_and_captures_by_zone.csv")
    starting_date = "2026-01-01"
    ending_date = "2026-03-31"
    obtained = process_effort_and_captures_data_by_zone(effort_captures_data)
    expected_number_of_months = 3
    expected_number_of_column_zones = 3
    assert obtained.shape() == (expected_number_of_months, expected_number_of_column_zones)
