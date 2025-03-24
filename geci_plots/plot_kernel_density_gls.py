from geci_plots import np, plt
from geoambiental import PointArray, get_kernel_density_geographic
import matplotlib.pyplot as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker


from PIL import Image

import pandas as pd
import geopandas as gpd


def _plot_kernel_density_gls(
    gls_data_path, global_shapefile_data_path, path_rose_wind, selected_contour, result_map_path
):
    hot = mpl.colormaps["hot_r"]
    new_hot = hot(np.linspace(0, 1, 256))
    new_hot[0:15, 3] = 0
    new_hot[15:-1, 3] = 0.3
    new_hor_r = ListedColormap(new_hot)

    colors = [
        (0.1, 0.1, 0.5, 0),
        (1, 0, 0, 0.5),
        (1, 1, 1, 1),
    ]

    albatros_gls_data = pd.read_csv(gls_data_path)
    albatros_gls_data.rename(
        columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True
    )
    global_shapefile = gpd.read_file(global_shapefile_data_path)
    rose_wind = Image.open(path_rose_wind)

    global_shapefile_translated = global_shapefile.translate(-360)
    mask_datos_trand = albatros_gls_data.longitude > 0
    albatros_gls_data.loc[mask_datos_trand, "longitude"] = (
        albatros_gls_data[mask_datos_trand]["longitude"] - 360
    )

    point_array = PointArray(albatros_gls_data["latitude"], albatros_gls_data["longitude"])
    kernel = get_kernel_density_geographic(point_array, bandwidth=0.04)
    normalized_kernel = np.array(kernel[2]) / np.nanmax(kernel[2])

    fig, ax = plt.subplots(figsize=(14.3, 10.4))
    land_color = "#FFFAE6"
    global_shapefile.plot(ax=ax, color=land_color, edgecolor="black", linewidth=0.3)
    global_shapefile_translated.plot(ax=ax, color=land_color, edgecolor="black", linewidth=0.3)

    plt.plot(albatros_gls_data["longitude"], albatros_gls_data["latitude"], ".b", markersize=3)
    if selected_contour == "All_contours":
        plt.contourf(kernel[0], kernel[1], normalized_kernel, 100, cmap=new_hor_r)
    elif selected_contour == "50_contour":
        plt.contourf(
            kernel[0],
            kernel[1],
            normalized_kernel,
            [0, np.max(normalized_kernel) / 2, np.max(normalized_kernel)],
            colors=colors,
        )

    sea_color = "#E6FFFF"
    plt.gca().set_facecolor(sea_color)
    plt.xlim(-180, -95)
    plt.ylim(10, 65)
    plt.yticks(size=20)
    plt.xticks(size=20)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))

    img_x = 1150
    img_y = 730

    width, height = rose_wind.size
    rescale_factor = 4
    new_size = (round(width / rescale_factor), round(height / rescale_factor))
    resized_rose_wind = rose_wind.resize(new_size)
    fig.figimage(resized_rose_wind, img_x, img_y, zorder=100)
    plt.savefig(result_map_path)
