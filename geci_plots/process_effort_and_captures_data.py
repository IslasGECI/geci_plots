from geci_plots.geci_plots import (
    select_date_interval,
    sort_monthly_dataframe,
    generate_monthly_ticks,
    geci_plot,
    annotated_bar_plot_by_columns,
    plot_points_with_labels,
)
from geci_plots.process_cameras_data import summarize_monthly_cameras_effort_and_captures

import matplotlib.pyplot as plt


def _plot_monthly_traps_effort_and_captures_by_zone(effort_captures_data, begin_date, final_date):
    processed_effort_captures_by_zone = process_effort_and_captures_data_by_zone(
        effort_captures_data, begin_date, final_date
    )
    effort_data = summarize_monthly_cameras_effort_and_captures(
        effort_captures_data, begin_date, final_date
    )

    x_ticks = generate_monthly_ticks(processed_effort_captures_by_zone)
    fontsize = 20

    _, ax = geci_plot()
    annotated_bar_plot_by_columns(
        ax, processed_effort_captures_by_zone, x_ticks, y_pos=1, fontsize=fontsize
    )
    ax.set_ylabel("Captures per month (No. cats dispatched)", fontsize=fontsize)
    ax2 = ax.twinx()
    plot_points_with_labels(ax2, effort_data, x_ticks, column_key="Effort", y_pos=1500)
    ax2.set_ylabel("Effort per month (night traps)", fontsize=fontsize)
    ax2.tick_params(axis="both", labelsize=fontsize, labelrotation=90)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.tight_layout()
    return ax, ax2


def process_effort_and_captures_data_by_zone(effort_captures_data, begin_date, final_date):
    effort_captures_by_zone = effort_captures_data.copy()
    data_sorted = sort_monthly_dataframe(effort_captures_by_zone, date_format="ISO-8601")
    pivot_table = data_sorted.pivot_table("Captures", "Date", "Zone", fill_value=0)
    pivot_table_filled = pivot_table.asfreq("MS").fillna(0)
    pivot_table_filled = select_date_interval(pivot_table_filled, begin_date, final_date)

    columns = pivot_table_filled.keys()
    new_columns_names = ["Zone {}".format(name) for name in columns]
    pivot_table_filled.columns = new_columns_names
    return pivot_table_filled
