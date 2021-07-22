import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from pandas._testing import assert_frame_equal
from geci_plots import (
    create_box_plot_data,
    historic_mean_effort,
    filter_by_season_and_zone,
    ticks_positions_array,
    roundup,
    order_magnitude,
    rounded_ticks_array,
    heatmap,
    annotate_heatmap,
)
random_state = np.random.RandomState(1)

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
    assert_frame_equal(test_for_data, obtained_data)


def test_ticks_positions_array():
    test_tick = [1, 2, 3]
    expected_tick_position = np.array([1.0, 2.0, 3.05])
    obtained_tick_position = ticks_positions_array(test_tick)
    np.testing.assert_equal(expected_tick_position, obtained_tick_position)


def test_roundup():
    expected_rounded = 20
    multiplier = 10
    number_to_be_rounded = 11
    obtained_rounded = roundup(number_to_be_rounded, multiplier)
    assert expected_rounded == obtained_rounded


def test_order_magnitude():
    test_for_data = np.array([1.0, 30.0, 60.0])
    expected_order_magnitude = np.array(1)
    obtained_order_magnitude = order_magnitude(test_for_data)
    np.testing.assert_equal(expected_order_magnitude, obtained_order_magnitude)


def test_rounded_ticks_array():
    superior_limit = 365
    min_value = 1
    expected_rounded_ticks_array = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    obtained_rounded_ticks_array = rounded_ticks_array(superior_limit, min_value)
    np.testing.assert_equal(expected_rounded_ticks_array, obtained_rounded_ticks_array)


@pytest.mark.mpl_image_compare(tolerance=0, savefig_kwargs={"dpi": 300})
def test_heatmap():
    data_to_plot = random_state.rand(5,5)
    x_labels = np.linspace(10,20,5)
    y_labels = np.linspace(10,20,5)
    fig, ax = plt.subplots()
    image, color_bar = heatmap(data_to_plot, x_labels, y_labels, 20, ax)
    texts = annotate_heatmap(image, valfmt="{x:.1f}", size=15)
    return fig