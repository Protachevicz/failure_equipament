
# failure_classification_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_excel("Test_O_G_Equipment_Data.xlsx")

# Define operational regimes
df["Regime"] = df["Preset_1"].astype(str) + "-" + df["Preset_2"].astype(str)

# Function to classify the likely root cause of failure
def classify_failure(row):
    if not row['Fail']:
        return 'Normal'
    if row['Temperature'] > 100 and row['VibrationY'] < 80:
        return 'Overheating'
    elif row['Pressure'] > 120 and row['VibrationY'] < 80:
        return 'Overpressure'
    elif max(row['VibrationX'], row['VibrationY'], row['VibrationZ']) > 120:
        return 'Excessive vibration'
    elif row['Frequency'] > 110:
        return 'Overspeed'
    else:
        return 'Other'

# Apply classification
df["Failure_Type"] = df.apply(classify_failure, axis=1)

# Create output directory
os.makedirs("figs", exist_ok=True)

# Plot 1: Distribution of inferred failure types
plt.figure(figsize=(10,6))
sns.countplot(data=df[df['Fail']], x="Failure_Type", hue="Failure_Type", palette="Set2", legend=False,
              order=df["Failure_Type"].value_counts().index)
plt.title("Distribution of Inferred Failure Types")
plt.ylabel("Number of Failure Cycles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figs/failure_type_distribution.png")

# Plot 2: Mean parameter values per failure type
param_cols = ["Temperature", "Pressure", "VibrationX", "VibrationY", "VibrationZ", "Frequency"]
df_fail = df[df['Fail']].groupby("Failure_Type")[param_cols].mean().reset_index()
melted = df_fail.melt(id_vars="Failure_Type", var_name="Parameter", value_name="Value")

plt.figure(figsize=(12,6))
sns.barplot(data=melted, x="Parameter", y="Value", hue="Failure_Type")
plt.title("Mean Parameter Values per Failure Type")
plt.ylabel("Mean Value")
plt.legend(title="Failure Type")
plt.tight_layout()
plt.savefig("figs/parameters_by_failure_type.png")

print("Figures saved to 'figs/' folder.")
