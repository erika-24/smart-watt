import pandas as pd
from typing import Optional, Dict

def get_load_cost_forecast(
    df_final: pd.DataFrame,
    method: Optional[str] = "hp_hc_periods",
    csv_path: Optional[str] = "data_load_cost_forecast.csv",
    peak_hours: Optional[Dict[str, Dict[str, str]]] = None,
    offpeak_cost: Optional[float] = None,
    peak_cost: Optional[float] = None,
    list_and_perfect: Optional[bool] = False
) -> pd.DataFrame:
    """
    Get the unit cost for load consumption based on multiple tariff periods.

    :param df_final: DataFrame containing input data.
    :param method: Method for load cost forecast ('hp_hc_periods' or 'csv'), defaults to 'hp_hc_periods'.
    :param csv_path: Path to CSV file if using 'csv' method.
    :param peak_hours: Dictionary defining peak hour periods, e.g., {"morning": {"start": "07:00", "end": "10:00"}}.
    :param offpeak_cost: Cost for off-peak hours.
    :param peak_cost: Cost for peak hours.
    :param list_and_perfect: Optional flag for additional processing, defaults to False.
    :return: DataFrame with appended load cost column.
    """

    if method == "hp_hc_periods" and peak_hours and offpeak_cost is not None and peak_cost is not None:
        df_final["load_cost"] = None  # Initialize without setting default cost
        for _, period in peak_hours.items():
            peak_indices = df_final.between_time(period["start"], period["end"]).index
            df_final.loc[peak_indices, "load_cost"] = peak_cost
        df_final["load_cost"].fillna(offpeak_cost, inplace=True)  # Only fill missing values if offpeak_cost is provided

    elif method == "csv":
        df_csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df_final = df_final.merge(df_csv, left_index=True, right_index=True, how="left")

    return df_final
