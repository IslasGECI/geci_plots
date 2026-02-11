from geci_plots.geci_plots import select_date_interval

import pandas as pd


def summarize_monthly_cameras_effort_and_captures(cameras_data, begin_date, final_date):
    cameras_data["Date"] = pd.to_datetime(cameras_data["Date"])
    cameras_data = cameras_data.set_index(["Date"]).sort_index()
    cameras_data = select_date_interval(cameras_data, begin_date, final_date)
    cameras_data = cameras_data.resample("MS").sum()
    return cameras_data
