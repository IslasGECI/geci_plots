from geci_plots import plt
import matplotlib.dates as mdates
import pandas as pd


def plot_population_time_series(data):
    fig, ax = plt.subplots(figsize=(14.3, 10.4))

    data["Date"] = pd.to_datetime(data["Date"])

    ax.plot(data["Date"], data["Captures"])

    plt.ylabel("Number of individuals")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=90)

    return ax
