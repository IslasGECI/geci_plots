from geci_plots import plt
from geci_plots.plot_kernel_density_gls import (
    format_plot,
    plot_global_politic_division,
    plot_geographic_points,
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
        alpha=0.3,
    )
    plot_windrose(path_rose_wind, fig)
    return ax
