from geci_plots.vegetation_coverage import transform_quadrants_vegetation_data

import pandas as pd


def test_transform_quadrants_vegetation_data():
    df = pd.read_csv("tests/data/quadrants_vegetation_sampling.csv")
    obtained_df = transform_quadrants_vegetation_data(df)
    expected_shape = (8, 4)
    assert obtained_df.shape == expected_shape
