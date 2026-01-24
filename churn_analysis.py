import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# DAY 1: DATA LOADING & EDA
# =========================

# Load dataset
df = pd.read_csv("TelcoCustomerChurnDataFile.csv")

print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nData types:\n", df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# -------------------------
# EDA
# -------------------------
sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x="ChurnLabel", data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Contract", hue="ChurnLabel", data=df)
plt.title("Churn vs Contract Type")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="TenureinMonths", hue="ChurnLabel", bins=30, kde=True)
plt.title("Churn vs Tenure")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="ChurnLabel", y="MonthlyCharge", data=df)
plt.title("Churn vs Monthly Charges")
plt.show()

# =========================
# DAY 2: DATA CLEANING
# =========================

df_clean = df.copy()

TARGET = "ChurnLabel"

# Remove leakage columns
leakage_cols = [
    "CustomerID",
    "CustomerStatus",
    "ChurnScore",
    "ChurnCategory",
    "ChurnReason"
]

df_clean.drop(columns=leakage_cols, inplace=True)

print("\nColumns after removing leakage:")
print(df_clean.columns)

# Convert Yes/No feature columns ONLY (exclude target)
yes_no_cols = [
    col for col in df_clean.select_dtypes(include="object").columns
    if col != TARGET
]

for col in yes_no_cols:
    if set(df_clean[col].dropna().unique()) <= {"Yes", "No"}:
        df_clean[col] = df_clean[col].map({"Yes": 1, "No": 0})

print("\nConverted Yes/No feature columns to numeric")

# Separate features and target
X = df_clean.drop(TARGET, axis=1)
y = df_clean[TARGET].map({"Yes": 1, "No": 0})

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nMissing values (top columns):")
print(X.isnull().sum().sort_values(ascending=False).head())

# =========================
# DAY 3: FEATURE ENGINEERING
# =========================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Identify feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Train-test split (STRATIFIED)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# =========================
# DAY 4: MODEL TRAINING
# =========================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Train
log_reg.fit(X_train_processed, y_train)

# Predict
y_pred_lr = log_reg.predict(X_test_processed)

# Evaluate
print("\n--- Logistic Regression Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall   :", recall_score(y_test, y_pred_lr))
print("F1-score :", f1_score(y_test, y_pred_lr))

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train
rf_model.fit(X_train_processed, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test_processed)

# Evaluate
print("\n--- Random Forest Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1-score :", f1_score(y_test, y_pred_rf))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
