import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load data
df = pd.read_excel("Test_O_G_Equipment_Data.xlsx")

# Create a column to predict failure in the next cycle
df["Fail_next"] = df["Fail"].shift(-1)
df = df.dropna().copy()

# Convert to boolean
df["Fail"] = df["Fail"].astype(bool)
df["Fail_next"] = df["Fail_next"].astype(bool)

# Define safe sizes for validation
fail_all = df[df["Fail_next"] == True]
ok_all = df[df["Fail_next"] == False]

# Adapt number of samples for validation set
n_val_each = min(15, len(fail_all), len(ok_all))
if n_val_each < 5:
    raise ValueError(f"Not enough positive/negative samples for reliable validation. (Only {n_val_each})")

# Split validation sets
fail_val = fail_all.sample(n=n_val_each, random_state=42)
ok_val = ok_all.sample(n=n_val_each, random_state=42)
df_val = pd.concat([fail_val, ok_val])

# Training sets
fail_train = fail_all.drop(index=fail_val.index)
ok_train = ok_all.drop(index=ok_val.index)
df_train = pd.concat([fail_train, ok_train])

# Define features
features = ["Preset_1", "Preset_2", "Temperature", "Pressure",
            "VibrationX", "VibrationY", "VibrationZ", "Frequency"]
X_train = df_train[features]
y_train = df_train["Fail_next"].astype(int)
X_val = df_val[features]
y_val = df_val["Fail_next"].astype(int)

# Dataset info
print("\n[INFO] Dataset overview:")
print(f"- Training set: {len(df_train)} samples "
      f"({y_train.value_counts()[1]} with failure, {y_train.value_counts()[0]} without failure)")
print(f"- Validation set: {len(df_val)} samples "
      f"({y_val.value_counts()[1]} with failure, {y_val.value_counts()[0]} without failure)")

# Cross-validation setup
n_splits = min(5, max(2, y_train.value_counts().min()))
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Models
models = {
    "LogisticRegression + Scaler": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "GradientBoosting": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1))
    ])
}

# Evaluation loop
for name, model in models.items():
    print(f"\n======================== {name} ========================")

    try:
        y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv, method="predict")
        print("\n>> Cross-Validation Performance (Training):")
        print("Confusion Matrix:")
        print(confusion_matrix(y_train, y_pred_cv))
        print("Classification Report:")
        print(classification_report(y_train, y_pred_cv))
    except Exception as e:
        print(f"[Cross-validation error] {e}")

    # Fit on training data
    model.fit(X_train, y_train)

    # Evaluate on hold-out validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    print("\n>> Hold-Out Validation Performance:")
    print(f"ROC AUC: {roc_auc_score(y_val, y_val_proba):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred))

