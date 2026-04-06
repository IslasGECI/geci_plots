from geci_plots import plt
import matplotlib.dates as mdates
import pandas as pd


def plot_population_time_series(data):
    _, ax = plt.subplots(figsize=(14.3, 10.4))

    data["Date"] = pd.to_datetime(data["Date"])

    ax.plot(data["Date"], data["Captures"], marker="o", label="Captures")
    ax.fill_between(data["Date"], data["population_size"], data["population_size_percentile_95"])

    fontsize = 15
    plt.ylabel("Number of individuals", fontsize=fontsize + 5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    loc, labels = plt.xticks()
    plt.xticks(loc[1:-1], labels[1:-1], rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize="xx-large")

    return ax
