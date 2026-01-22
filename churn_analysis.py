import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("TelcoCustomerChurnDataFile.csv")

# Basic checks
print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nData types:\n", df.dtypes)

# View first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# View last 5 rows
print("\nLast 5 rows:")
print(df.tail())



sns.set(style="whitegrid")

# Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='ChurnLabel', data=df)
plt.title("Churn Distribution")
plt.show()

# Churn vs Contract
plt.figure(figsize=(6,4))
sns.countplot(x='Contract', hue='ChurnLabel', data=df)
plt.title("Churn vs Contract Type")
plt.show()

# Churn vs Tenure
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='TenureinMonths', hue='ChurnLabel', bins=30, kde=True)
plt.title("Churn vs Tenure")
plt.show()

# Churn vs Monthly Charges
plt.figure(figsize=(6,4))
sns.boxplot(x='ChurnLabel', y='MonthlyCharge', data=df)
plt.title("Churn vs Monthly Charges")
plt.show()

# =========================
# DAY 2: DATA CLEANING
# =========================

# Make a copy of original data
df_clean = df.copy()

# Target column
TARGET = "ChurnLabel"

# Columns that cause data leakage (MUST REMOVE)
leakage_cols = [
    "CustomerID",
    "CustomerStatus",
    "ChurnScore",
    "ChurnCategory",
    "ChurnReason"
]

# Drop leakage columns
df_clean.drop(columns=leakage_cols, inplace=True)

print("Columns after removing leakage:")
print(df_clean.columns)

# Convert Yes/No columns to 1/0
yes_no_cols = df_clean.select_dtypes(include='object').columns

for col in yes_no_cols:
    if set(df_clean[col].unique()) <= {"Yes", "No"}:
        df_clean[col] = df_clean[col].map({"Yes": 1, "No": 0})

print("Converted Yes/No columns to numeric")

# Separate features and target
X = df_clean.drop(TARGET, axis=1)
y = df_clean[TARGET].map({"Yes": 1, "No": 0})

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nMissing values:")
print(X.isnull().sum().sort_values(ascending=False).head())
