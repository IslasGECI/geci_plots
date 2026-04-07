from geci_plots.geci_plots import geci_plot

import matplotlib.pyplot as plt


def plot_histogram_with_limits(x, bins, limits=[], plot_options={}, lines_options={}):
    _, ax = geci_plot()
    plt.xlabel(plot_options.pop("label", None), fontsize=plot_options.pop("fontsize", None))

    ax.hist(x, bins=bins, **plot_options)
    for lines in limits:
        ax.axvline(x=lines, **lines_options)
    return ax
