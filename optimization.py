import pandas as pd
import pulp as plp
import logging

class Optimization:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.optim_status = None

    def perform_optimization(self, data_opt: pd.DataFrame) -> pd.DataFrame:
        """
        Perform the optimization using linear programming (LP).

        :param data_opt: Input DataFrame with energy consumption and production data
        :return: Optimized DataFrame with results
        """
        P_PV = data_opt['P_PV'].values
        P_Load = data_opt['P_Load'].values
        unit_load_cost = data_opt['unit_load_cost'].values
        unit_prod_price = data_opt['unit_prod_price'].values
        
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
            unit_load_cost[i] * P_Load[i] + unit_prod_price[i] * P_grid_neg[i] 
            for i in set_I
        )
        opt_model.setObjective(objective)

        # Constraints: Balance between power production, consumption, and grid power
        for i in set_I:
            opt_model += P_PV[i] - P_deferrable[i] - P_Load[i] + P_grid_neg[i] + P_grid_pos[i] >= -0.01, f"Power_balance_{i}"

        # Solve the problem
        opt_model.solve()

        self.optim_status = plp.LpStatus[opt_model.status]
        self.logger.info(f"Optimization status: {self.optim_status}")

        # Collect results
        opt_results = pd.DataFrame()
        opt_results["P_PV"] = P_PV
        opt_results["P_Load"] = P_Load
        opt_results["P_grid_pos"] = [P_grid_pos[i].varValue for i in set_I]
        opt_results["P_grid_neg"] = [P_grid_neg[i].varValue for i in set_I]
        opt_results["P_deferrable"] = [P_deferrable[i].varValue for i in set_I]
        opt_results["total_cost"] = [unit_load_cost[i] * P_Load[i] + unit_prod_price[i] * P_grid_neg[i] for i in set_I]

        return opt_results

# Initialize logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load sample energy data
data_file = "sample_energy_data.csv"
df_energy_data = pd.read_csv(data_file)

# Initialize the optimization module
optim = Optimization(logger)

# Perform optimization
optimized_results = optim.perform_optimization(df_energy_data)

# Save results to CSV
optimized_results.to_csv("optimized_energy_data.csv", index=False)

print("Optimization completed. Results saved to optimized_energy_data.csv.")