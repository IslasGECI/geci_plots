import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from pandas._testing import assert_frame_equal
from geci_plots import (
    annotate_heatmap,
    annotate_pie_chart,
    annotated_bar_plot,
    calculate_values_for_age_pie_chart,
    calculate_values_for_sex_pie_chart,
    create_box_plot_data,
    filter_by_season_and_zone,
    geci_plot,
    generate_monthly_ticks,
    heatmap,
    historic_mean_effort,
    order_magnitude,
    plot_comparative_annual_effort_by_zone,
    plot_points_with_labels,
    prepare_cats_by_zone_and_age,
    prepare_cats_by_zone_and_sex,
    rounded_ticks_array,
    roundup,
    select_date_interval,
    sort_monthly_dataframe,
    ticks_positions_array,
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
    data_to_plot = random_state.rand(5, 5)
    x_labels = np.linspace(10, 20, 5)
    y_labels = np.linspace(10, 20, 5)
    fig, ax = plt.subplots()
    image, color_bar = heatmap(data_to_plot, x_labels, y_labels, 20, ax)
    annotate_heatmap(image, valfmt="{x:.1f}", size=15)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, savefig_kwargs={"dpi": 300})
def test_calculate_values_for_age_pie_chart():
    data_ages = pd.read_csv("tests/data/annual_age_data.csv")
    data_ages = data_ages.dropna()
    data_ages = prepare_cats_by_zone_and_age(data_ages)
    season = 2021
    fig, ax = plt.subplots()
    pie_values, pie_labels = calculate_values_for_age_pie_chart(data_ages, season)
    wedges_zones, texts = ax.pie(pie_values.sum(axis=1))
    annotate_pie_chart(ax, wedges_zones, pie_labels)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, savefig_kwargs={"dpi": 300})
def test_calculate_values_for_sex_pie_chart():
    data_ages = pd.read_csv("tests/data/annual_sex_data.csv")
    data_ages = data_ages.dropna()
    data_ages = prepare_cats_by_zone_and_sex(data_ages)
    season = 2021
    fig, ax = plt.subplots()
    pie_values, pie_labels = calculate_values_for_sex_pie_chart(data_ages, season)
    wedges_zones, texts = ax.pie(pie_values.sum(axis=1))
    annotate_pie_chart(ax, wedges_zones, pie_labels)
    return fig


@pytest.mark.mpl_image_compare(tolerance=0, savefig_kwargs={"dpi": 300})
def test_plot_comparative_annual_effort_by_zone():
    data_captures = pd.read_csv("tests/data/annual_captures_data.csv")
    data_captures = data_captures[data_captures["Season"].isin([2020, 2021])]
    fig, ax = geci_plot()
    plot_comparative_annual_effort_by_zone(ax, data_captures, fontsize=25, bar_label_size=17)
    return fig


def test_sort_monthly_dataframe():
    expected_sorted_dataframe = np.array([[3], [2], [4], [5]])

    dataframe = pd.DataFrame(
        {"Date": ["1995/Jun", "1995/May", "1995/Jul", "1995/Dic"], "value": [2, 3, 4, 5]}
    )
    obtained_sorted_dataframe = sort_monthly_dataframe(dataframe)

    np.testing.assert_equal(expected_sorted_dataframe, obtained_sorted_dataframe)


@pytest.mark.mpl_image_compare(tolerance=0, savefig_kwargs={"dpi": 300})
def test_annotated_bar_plot():
    data_captures = pd.read_csv("tests/data/monthly_data.csv")
    data_captures = sort_monthly_dataframe(data_captures, date_format="ISO-8601")
    data_captures = data_captures.resample("MS").sum()
    data_captures = select_date_interval(data_captures, "2021-01-01")
    x_ticks = generate_monthly_ticks(data_captures)
    fig, ax = geci_plot()
    annotated_bar_plot(ax, data_captures, x_ticks, column_key="Captures", y_pos=1)
    ax2 = ax.twinx()
    plot_points_with_labels(ax2, data_captures, x_ticks, column_key="Effort", y_pos=1500)
    return fig
