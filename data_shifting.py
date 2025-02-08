import pandas as pd

# Load the training data
csv_path = "training_data_load.csv"
df_load = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

df_load.index = df_load.index.tz_localize("UTC") if df_load.index.tz is None else df_load.index.tz_convert("UTC")

# Define the new start date as timezone-aware
new_start_date = pd.Timestamp("2024-02-08 07:00:00", tz="UTC")

# Calculate the time shift
original_start_date = df_load.index[0]  # First timestamp in dataset
time_shift = new_start_date - original_start_date  # Time difference

# Apply the shift to all timestamps
df_load.index = df_load.index + time_shift

# Save the adjusted data back to CSV (optional)
df_load.to_csv("training_load_data_shifted.csv")

# Confirm the changes
print(df_load.head())  # Check first few rows
