
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("Test_O_G_Equipment_Data.xlsx")
df["Fail"] = df["Fail"].astype(bool)

# Group by Preset_1 and Preset_2
preset_failures = df.groupby(["Preset_1", "Preset_2"]).agg(
    total_cycles=("Fail", "count"),
    failure_count=("Fail", "sum")
).reset_index()

# Calculate failure rate
preset_failures["failure_rate"] = preset_failures["failure_count"] / preset_failures["total_cycles"]

# Heatmap of failure rates
plt.figure(figsize=(10, 6))
pivot = preset_failures.pivot(index="Preset_1", columns="Preset_2", values="failure_rate")
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", cbar_kws={'label': 'Failure Rate'})
plt.title("Failure Rate by Preset_1 and Preset_2 Configuration")
plt.xlabel("Preset_2")
plt.ylabel("Preset_1")
plt.tight_layout()
plt.show()

# Print failure rate table
print(preset_failures.sort_values("failure_rate", ascending=False))
