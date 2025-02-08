import pickle
import pandas as pd

# Define the file paths
pkl_file_path = "data_train_load_clustering.pkl"  # Update this if needed
csv_file_path = "converted_data.csv"  # Output file

# Step 1: Load the Pickle File
try:
    with open(pkl_file_path, "rb") as file:
        data = pickle.load(file)
except FileNotFoundError:
    print(f"Error: File '{pkl_file_path}' not found. Check the file path.")
    exit()
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

# Step 2: Inspect the Pickle Contents
print(f"✅ Successfully loaded '{pkl_file_path}'")
print(f"🔍 Data type: {type(data)}")

# If it's a tuple, unpack and inspect elements
if isinstance(data, tuple):
    print(f"Tuple contains {len(data)} elements.")
    for i, item in enumerate(data):
        print(f"\nElement {i}: Type -> {type(item)}")
        if isinstance(item, pd.DataFrame):
            df = item
            print("📊 Found a DataFrame! Preview:")
            print(df.head())
            break  # Stop at the first found DataFrame
    else:
        print("❌ No DataFrame found in the tuple.")
        exit()
elif isinstance(data, pd.DataFrame):
    df = data  # Directly assign if data is already a DataFrame
    print("📊 Found a DataFrame! Preview:")
    print(df.head())
else:
    print("❌ Data is not a DataFrame or a tuple containing a DataFrame.")
    exit()

# Step 3: Convert DataFrame to CSV
try:
    df.to_csv(csv_file_path, index=True)
    print(f"✅ Successfully converted to CSV: '{csv_file_path}'")
except Exception as e:
    print(f"❌ Error saving CSV: {e}")
