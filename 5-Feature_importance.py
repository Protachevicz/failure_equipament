import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Carrega os dados
df = pd.read_excel("Test_O_G_Equipment_Data.xlsx")

# Cria coluna para prever falha no próximo ciclo
df["Fail_next"] = df["Fail"].shift(-1)
df = df.dropna().copy()

# Converte para booleano
df["Fail"] = df["Fail"].astype(bool)
df["Fail_next"] = df["Fail_next"].astype(bool)

# Filtra apenas ciclos normais
df = df[df["Fail"] == False].copy()

# Features e alvo
features = ["Preset_1", "Preset_2", "Temperature", "Pressure",
            "VibrationX", "VibrationY", "VibrationZ", "Frequency"]
X = df[features]
y = df["Fail_next"].astype(int)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1)
gb_model.fit(X, y)

# Importância das features - Gradient Boosting
importances_gb = pd.Series(gb_model.feature_importances_, index=features).sort_values(ascending=False)

# Plot - Gradient Boosting
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_gb.values, y=importances_gb.index, palette="viridis")
plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance_gb.png")
plt.show()

# Logistic Regression com StandardScaler
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
])
pipeline_lr.fit(X, y)

# Importância das features - Logistic Regression (coeficientes absolutos)
coeffs = pipeline_lr.named_steps['clf'].coef_[0]
importances_lr = pd.Series(abs(coeffs), index=features).sort_values(ascending=False)

# Plot - Logistic Regression
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_lr.values, y=importances_lr.index, palette="magma")
plt.title("Feature Coefficients - Logistic Regression")
plt.xlabel("Absolute Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance_lr.png")
plt.show()

