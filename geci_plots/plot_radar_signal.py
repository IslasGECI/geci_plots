from geci_plots import plt
from geci_plots.plot_kernel_density_gls import (
    format_plot,
    plot_global_politic_division,
    plot_windrose,
)


def plot_radar_signal_geographic_points(
    radar_signal_points_data, global_shapefile_data_path, path_rose_wind
):
    fig, ax = plt.subplots(figsize=(14.3, 10.4))
    format_plot(ax, radar_signal_points_data)
    plot_global_politic_division(global_shapefile_data_path, ax)
    plt.scatter(
        radar_signal_points_data["longitude"],
        radar_signal_points_data["latitude"],
        c=radar_signal_points_data["radar_signal"],
        alpha=0.9,
        cmap="cool",
        s=radar_signal_points_data["radar_signal"] * 0.5,
        norm="log",
    )
    plot_windrose(path_rose_wind, fig)
    cbar = plt.colorbar(ax=ax, location="left")
    fontsize = 20
    cbar.set_label("Radar Signal Strength", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    return ax
