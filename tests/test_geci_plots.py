import numpy as np
import pandas as pd
from geci_plots import (
    create_box_plot_data,
    historic_mean_effort,
    filter_by_season_and_zone,
    ticks_positions_array,
)


def test_create_box_plot_data():
    df_test = pd.DataFrame({"Temporada": [2000, 2001, 2002, 2001], "Longitud": [10, 20, 30, 40]})
    obtained_box_plot_data, obtained_seasons = create_box_plot_data(df_test, "Longitud")
    expected_box_plot_data = [
        pd.Series(10, index=[0], name="Longitud"),
        pd.Series([20, 40], index=[1, 3], name="Longitud"),
        pd.Series(30, index=[2], name="Longitud"),
    ]
    expected_seasons = [2000, 2001, 2002]
    np.testing.assert_array_equal(obtained_seasons, expected_seasons)
    for i in range(3):
        pd.testing.assert_series_equal(obtained_box_plot_data[i], expected_box_plot_data[i])


def test_historic_mean_effort():
    test_for_data = {"Season": np.array([1, 1, 1, 1]), "Zone": [2, 2, 2, 2]}
    obtained_mean = historic_mean_effort(test_for_data, "Season")
    expected_mean = 1.0
    assert expected_mean == obtained_mean


def test_filter_by_season_and_zone():
    test_for_data = pd.DataFrame({"Season": np.array([1, 1, 1, 1]), "Zone": [2, 2, 2, 2]})
    obtained_data = filter_by_season_and_zone(test_for_data, 1, 2)


def test_ticks_positions_array():
    test_tick = [1, 2, 3]
    expected_tick_position = np.array([1.0, 2.0, 3.05])
    obtained_tick_position = ticks_positions_array(test_tick)
    np.testing.assert_equal(expected_tick_position, obtained_tick_position)
