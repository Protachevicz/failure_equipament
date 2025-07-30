import pandas as pd

# Load data
df = pd.read_excel("Test_O_G_Equipment_Data.xlsx")

# Ensure 'Fail' is boolean
df["Fail"] = df["Fail"].astype(bool)

# Count every individual failure (every True value)
total_failures = df["Fail"].sum()

print(f"Total number of failures (each line with Fail=True counts): {total_failures}")

