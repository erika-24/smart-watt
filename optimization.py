import logging
import pulp as plp
import pandas as pd
import numpy as np

class Optimization:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.optim_status = None

    def perform_optimization(self, data_opt: pd.DataFrame, P_PV: np.array, P_load: np.array, unit_load_cost: np.array, unit_prod_price: np.array) -> pd.DataFrame:
        """
        Perform the optimization using linear programming (LP).

        :param data_opt: Input DataFrame with energy consumption and production data
        :param P_PV: Photovoltaic power production values
        :param P_load: Load power consumption values
        :param unit_load_cost: Cost for consuming energy per unit time
        :param unit_prod_price: Price for producing energy per unit time
        :return: Optimized DataFrame with results
        """
        # Basic settings
        opt_model = plp.LpProblem("EnergyOptimization", plp.LpMinimize)
        n = len(data_opt.index)  # Number of time steps
        set_I = range(n)

        # Decision variables
        P_grid_pos = {i: plp.LpVariable(f"P_grid_pos_{i}", lowBound=0) for i in set_I}  # Power from grid
        P_grid_neg = {i: plp.LpVariable(f"P_grid_neg_{i}", lowBound=0) for i in set_I}  # Power to grid
        P_deferrable = {i: plp.LpVariable(f"P_deferrable_{i}", lowBound=0) for i in set_I}  # Deferrable load

        # Objective: Minimize total cost (consumption cost and production cost)
        objective = plp.lpSum(
            unit_load_cost[i] * P_load[i] + unit_prod_price[i] * P_grid_neg[i] 
            for i in set_I
        )
        opt_model.setObjective(objective)

        # Constraints: Balance between power production, consumption, and grid power
        for i in set_I:
            opt_model += P_PV[i] - P_deferrable[i] - P_load[i] + P_grid_neg[i] + P_grid_pos[i] == 0, f"Power_balance_{i}"

        # Solve the problem
        opt_model.solve()

        self.optim_status = plp.LpStatus[opt_model.status]
        self.logger.info(f"Optimization status: {self.optim_status}")

        # Collect results
        opt_results = pd.DataFrame()
        opt_results["P_PV"] = P_PV
        opt_results["P_Load"] = P_load
        opt_results["P_grid_pos"] = [P_grid_pos[i].varValue for i in set_I]
        opt_results["P_grid_neg"] = [P_grid_neg[i].varValue for i in set_I]
        opt_results["P_deferrable"] = [P_deferrable[i].varValue for i in set_I]
        opt_results["total_cost"] = [unit_load_cost[i] * P_load[i] + unit_prod_price[i] * P_grid_neg[i] for i in set_I]

        return opt_results

# Sample usage
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Sample data for testing
data = {
    'P_PV': np.array([5, 10, 15, 10, 5]),  # PV production in kW
    'P_Load': np.array([7, 7, 7, 7, 7]),  # Load consumption in kW
    'unit_load_cost': np.array([0.1, 0.1, 0.1, 0.1, 0.1]),  # Cost per unit for load
    'unit_prod_price': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # Price per unit for production
}
df_input_data = pd.DataFrame(data)

# Initialize the optimization object and perform optimization
optim = Optimization(logger)
results = optim.perform_optimization(df_input_data, df_input_data['P_PV'].values, df_input_data['P_Load'].values, df_input_data['unit_load_cost'].values, df_input_data['unit_prod_price'].values)

# Display the results
print(results)
