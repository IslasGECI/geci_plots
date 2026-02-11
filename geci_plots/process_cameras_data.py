from geci_plots.geci_plots import (
    select_date_interval,
    geci_plot,
    plot_points_with_labels,
    annotated_bar_plot,
    generate_monthly_ticks,
    order_magnitude,
    roundup,
)

import matplotlib.pyplot as plt
import pandas as pd


def _plot_monthly_cameras_effort_and_captures(cameras_data, begin_date, final_date):
    processed_cameras_data = summarize_monthly_cameras_effort_and_captures(
        cameras_data, begin_date, final_date
    )
    x_ticks = generate_monthly_ticks(processed_cameras_data)
    fontsize = 20

    fig, ax = geci_plot()

    annotated_bar_plot(ax, processed_cameras_data, x_ticks, column_key="Total_individuals", y_pos=1)
    ax.set_ylabel("Detections per month (cats in photos)", fontsize=fontsize)
    ax2 = ax.twinx()
    plot_points_with_labels(ax2, processed_cameras_data, x_ticks, column_key="Effort", y_pos=100)
    ax2.set_ylim(
        0,
        roundup(
            processed_cameras_data["Effort"].max(),
            10 ** order_magnitude(processed_cameras_data["Effort"]),
        ),
    )
    ax2.set_ylabel("Effort per month (camera night traps)", fontsize=fontsize)
    ax2.tick_params(axis="both", labelsize=fontsize, labelrotation=90)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.tight_layout()
    return ax


def summarize_monthly_cameras_effort_and_captures(cameras_data, begin_date, final_date):
    cameras_data["Date"] = pd.to_datetime(cameras_data["Date"])
    cameras_data = cameras_data.set_index(["Date"]).sort_index()
    cameras_data = select_date_interval(cameras_data, begin_date, final_date)
    cameras_data = cameras_data.resample("MS").sum()
    return cameras_data
