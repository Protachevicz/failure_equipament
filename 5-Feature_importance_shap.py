import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Carrega os dados
df = pd.read_excel("Test_O_G_Equipment_Data.xlsx")

# Prepara coluna para prever falha no próximo ciclo
df["Fail_next"] = df["Fail"].shift(-1)
df = df.dropna().copy()

# Converte para booleano
df["Fail"] = df["Fail"].astype(bool)
df["Fail_next"] = df["Fail_next"].astype(bool)

# Seleciona apenas ciclos normais para prever próxima falha
df = df[df["Fail"] == False].copy()

# Define features e alvo
features = ["Preset_1", "Preset_2", "Temperature", "Pressure",
            "VibrationX", "VibrationY", "VibrationZ", "Frequency"]
X = df[features]
y = df["Fail_next"].astype(int)

# --------------------- GRADIENT BOOSTING ---------------------
gb_model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1)
gb_model.fit(X, y)

# Importância do modelo padrão
importances_gb = pd.Series(gb_model.feature_importances_, index=features).sort_values(ascending=False)

# Plot da importância padrão
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_gb.values, y=importances_gb.index, palette="viridis")
plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_gb.png")
plt.show()

# --------------------- LOGISTIC REGRESSION ---------------------
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
])
pipeline_lr.fit(X, y)

# Coeficientes absolutos
coeffs = pipeline_lr.named_steps['clf'].coef_[0]
importances_lr = pd.Series(abs(coeffs), index=features).sort_values(ascending=False)

# Plot da regressão logística
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_lr.values, y=importances_lr.index, palette="magma")
plt.title("Feature Coefficients - Logistic Regression")
plt.xlabel("Absolute Coefficient Value")
plt.tight_layout()
plt.savefig("feature_importance_lr.png")
plt.show()

# --------------------- SHAP VALUES ---------------------
# Usa TreeExplainer para Gradient Boosting
explainer = shap.Explainer(gb_model, X)
shap_values = explainer(X)

# Summary plot (global importances)
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.close()

# Bar plot de importância média absoluta
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar_plot.png")
plt.close()

print("SHAP plots saved as 'shap_summary_plot.png' and 'shap_bar_plot.png'")

