from geci_plots.geci_plots import geci_plot


def plot_histogram_with_limits(x, bins, limits=[], plot_options={}, lines_options={}):
    _, ax = geci_plot()
    ax.hist(x, bins=bins, **plot_options)
    for lines in limits:
        ax.axvline(x=lines, **lines_options)
    return ax
