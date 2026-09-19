# Healthcare Cost Analysis & High-Cost Patient Prediction
# Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

DATA_PATH = "healthcare_cost_analysis_dataset.csv"

df = pd.read_csv(DATA_PATH)

# -----------------------------
# 1. Data Quality Checks
# -----------------------------
print(df.head())
print(df.info())
print(df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# -----------------------------
# 2. Feature Engineering
# -----------------------------
age_bins = [0, 18, 35, 50, 65, 100]
age_labels = ["0-18", "19-35", "36-50", "51-65", "66+"]

df["Age_Band"] = pd.cut(
    df["Age"], bins=age_bins, labels=age_labels, include_lowest=True
)

df["Cost_Per_Day"] = np.where(
    df["Length_of_Stay_Days"] > 0,
    df["Total_Cost_USD"] / df["Length_of_Stay_Days"],
    df["Total_Cost_USD"]
)

high_cost_threshold = df["Total_Cost_USD"].quantile(0.75)
df["High_Cost_Flag"] = (
    df["Total_Cost_USD"] >= high_cost_threshold
).astype(int)

# -----------------------------
# 3. Executive KPIs
# -----------------------------
total_cost = df["Total_Cost_USD"].sum()
avg_cost = df["Total_Cost_USD"].mean()
median_cost = df["Total_Cost_USD"].median()
avg_los = df["Length_of_Stay_Days"].mean()
readmission_rate = df["Readmission_30_Days"].mean() * 100
high_cost_rate = df["High_Cost_Flag"].mean() * 100

print("\nEXECUTIVE KPIs")
print("Total Cost:", total_cost)
print("Average Cost:", avg_cost)
print("Median Cost:", median_cost)
print("Average LOS:", avg_los)
print("30-Day Readmission Rate:", readmission_rate)
print("High-Cost Patient Rate:", high_cost_rate)

# -----------------------------
# 4. Department Analysis
# -----------------------------
department_summary = (
    df.groupby("Department")
      .agg(
          Patients=("Patient_ID", "count"),
          Avg_Cost=("Total_Cost_USD", "mean"),
          Total_Cost=("Total_Cost_USD", "sum"),
          Avg_LOS=("Length_of_Stay_Days", "mean")
      )
      .sort_values("Avg_Cost", ascending=False)
)

print("\nDepartment Summary")
print(department_summary)

department_summary["Avg_Cost"].plot(kind="bar", figsize=(10, 5))
plt.title("Average Healthcare Cost by Department")
plt.ylabel("Average Cost (USD)")
plt.xlabel("Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Condition Analysis
# -----------------------------
condition_summary = (
    df.groupby("Primary_Condition")
      .agg(
          Patients=("Patient_ID", "count"),
          Avg_Cost=("Total_Cost_USD", "mean"),
          Total_Cost=("Total_Cost_USD", "sum")
      )
      .sort_values("Avg_Cost", ascending=False)
)

print("\nCondition Summary")
print(condition_summary)

condition_summary["Avg_Cost"].plot(kind="bar", figsize=(10, 5))
plt.title("Average Healthcare Cost by Primary Condition")
plt.ylabel("Average Cost (USD)")
plt.xlabel("Primary Condition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Insurance Analysis
# -----------------------------
insurance_summary = (
    df.groupby("Insurance_Type")
      .agg(
          Patients=("Patient_ID", "count"),
          Avg_Cost=("Total_Cost_USD", "mean"),
          Total_Cost=("Total_Cost_USD", "sum")
      )
      .sort_values("Avg_Cost", ascending=False)
)

print("\nInsurance Summary")
print(insurance_summary)

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Insurance_Type", y="Total_Cost_USD")
plt.title("Healthcare Cost Distribution by Insurance Type")
plt.xlabel("Insurance Type")
plt.ylabel("Total Cost (USD)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Length of Stay vs Cost
# -----------------------------
plt.figure(figsize=(9, 6))
sns.regplot(
    data=df,
    x="Length_of_Stay_Days",
    y="Total_Cost_USD",
    scatter_kws={"alpha": 0.35}
)
plt.title("Length of Stay vs Total Healthcare Cost")
plt.xlabel("Length of Stay (Days)")
plt.ylabel("Total Cost (USD)")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Readmission Analysis
# -----------------------------
readmission_summary = (
    df.groupby("Readmission_30_Days")
      .agg(
          Patients=("Patient_ID", "count"),
          Avg_Cost=("Total_Cost_USD", "mean"),
          Avg_LOS=("Length_of_Stay_Days", "mean")
      )
)

print("\nReadmission Summary")
print(readmission_summary)

department_readmission = (
    df.groupby("Department")["Readmission_30_Days"]
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)

department_readmission.plot(kind="bar", figsize=(10, 5))
plt.title("30-Day Readmission Rate by Department")
plt.ylabel("Readmission Rate (%)")
plt.xlabel("Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Correlation Analysis
# -----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(11, 8))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Random Forest Model
# -----------------------------
target = "High_Cost_Flag"

features = [
    "Age", "Gender", "Region", "Insurance_Type",
    "Primary_Condition", "Admission_Type", "Department",
    "Length_of_Stay_Days", "Procedures_Count",
    "Medication_Count", "Followup_Visits",
    "Readmission_30_Days", "Claim_Amount_USD"
]

X = df[features]
y = df[target]

categorical_features = [
    "Gender", "Region", "Insurance_Type",
    "Primary_Condition", "Admission_Type", "Department"
]

numeric_features = [
    "Age", "Length_of_Stay_Days", "Procedures_Count",
    "Medication_Count", "Followup_Visits",
    "Readmission_30_Days", "Claim_Amount_USD"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numeric", "passthrough", numeric_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nMODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 11. Feature Importance
# -----------------------------
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

feature_importance = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    .sort_values("Importance", ascending=False)
    .head(15)
)

print("\nTop Feature Importance")
print(feature_importance)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)
plt.title("Top Drivers of High-Cost Classification")
plt.tight_layout()
plt.show()

# -----------------------------
# 12. Export Summary Tables
# -----------------------------
department_summary.to_csv("department_cost_summary.csv")
condition_summary.to_csv("condition_cost_summary.csv")
insurance_summary.to_csv("insurance_cost_summary.csv")
readmission_summary.to_csv("readmission_summary.csv")
feature_importance.to_csv("high_cost_feature_importance.csv", index=False)

print("\nAnalysis completed and summary files exported.")
