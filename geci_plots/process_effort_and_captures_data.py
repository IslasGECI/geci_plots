from geci_plots.geci_plots import select_date_interval, sort_monthly_dataframe


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
