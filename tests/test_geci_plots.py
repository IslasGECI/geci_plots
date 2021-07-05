from geci_plots import *  # noqa
import numpy as np
import pandas as pd


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
