import pandas as pd
import pulp as plp
from typing import Optional, Dict

def optimize_load_cost_forecast(
    df_final: pd.DataFrame,
    peak_hours: Optional[Dict[str, Dict[str, str]]] = None,
    offpeak_cost: float = 0.1,
    peak_cost: float = 0.2
) -> pd.DataFrame:
    """
    Optimizes the unit cost for load consumption by scheduling loads to minimize total energy cost.

    :param df_final: DataFrame containing input data.
    :param peak_hours: Dictionary defining peak hour periods, e.g., {"morning": {"start": "07:00", "end": "10:00"}}.
    :param offpeak_cost: Cost for off-peak hours.
    :param peak_cost: Cost for peak hours.
    :return: DataFrame with an optimized load cost schedule.
    """

    # Initialize LP problem
    prob = plp.LpProblem("Minimize_Energy_Cost", plp.LpMinimize)

    # Create decision variables for each time slot
    df_final["load_var"] = [plp.LpVariable(f"load_{i}", lowBound=0) for i in df_final.index]

    # Define cost function
    cost_expr = []
    for i, row in df_final.iterrows():
        is_peak = any(row.between_time(period["start"], period["end"]).shape[0] > 0 for _, period in peak_hours.items())
        unit_cost = peak_cost if is_peak else offpeak_cost
        cost_expr.append(unit_cost * df_final.at[i, "load_var"])

    # Objective: Minimize total energy cost
    prob += plp.lpSum(cost_expr), "Total_Energy_Cost"

    # Solve the LP problem
    prob.solve(plp.PULP_CBC_CMD(msg=False))

    # Assign optimized load cost values
    df_final["optimized_load"] = [plp.value(var) for var in df_final["load_var"]]

    return df_final
