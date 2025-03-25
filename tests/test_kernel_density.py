from geci_plots.plot_kernel_density_gls import get_kernel_density

import pandas as pd
import numpy as np


def test_get_kernel_density():
    gls_data = pd.read_csv("tests/data/gls_albatros_tests.csv")
    obtained = get_kernel_density(gls_data)
    assert np.max(obtained[2]) < 3.95
    assert np.max(obtained[0]) < 179.8
    assert np.min(obtained[0]) > -179.54
    assert np.max(obtained[1]) < 79.8
    assert np.min(obtained[1]) > -19.44
