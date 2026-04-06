from geci_plots.plot_population_time_series import plot_population_time_series
import matplotlib as plt
import pandas as pd


def test_plot_population_time_series():
    data = pd.DataFrame(
        {
            "Date": [
                "2021-01-01",
                "2021-02-01",
                "2021-03-01",
                "2021-04-01",
                "2021-05-01",
                "2021-06-01",
                "2021-07-01",
                "2021-08-01",
                "2021-09-01",
                "2021-10-01",
                "2021-11-01",
                "2021-12-01",
                "2022-12-01",
            ],
            "population_size": [
                215.99034,
                217.26217,
                207.57934,
                204.766615,
                188.98587500000002,
                177.03384,
                178.93240500000002,
                180.825415,
                182.76378499999998,
                183.719325,
                180.631135,
                180.566105,
                180.566105,
            ],
            "population_size_percentile_95": [
                226.714813,
                228.05012,
                218.4337515,
                215.71285,
                199.96559,
                188.055699,
                190.0977185,
                192.154819,
                194.25333799999999,
                195.35084650000002,
                192.499587,
                192.783518,
                192.783518,
            ],
            "Births": [
                2.23487850679665,
                2.248038287605825,
                2.14784885944915,
                2.1187452493153374,
                1.9554599993946877,
                1.8317908291254,
                1.8514355137546126,
                1.8710227201183374,
                1.8910792719586624,
                1.9009663613923125,
                1.8690124811590374,
                1.8683396077828625,
                1.8683396077828625,
            ],
            "Captures": [1, 12, 5, 18, 14, 0, 0, 0, 1, 5, 2, 3, 2],
        }
    )
    obtained = plot_population_time_series(data)
    plt.pyplot.savefig("prueba.png")
    assert isinstance(obtained, plt.axes._axes.Axes)

    expected_ylabel = "Number of individuals"
    assert obtained.get_ylabel() == expected_ylabel
    assert obtained.get_xticklabels()[0].get_text() == "Jan 2021"
    assert obtained.get_xticklabels()[-1].get_text() == "Dec 2022"
    assert "o" in obtained.get_lines()[0].get_marker()
