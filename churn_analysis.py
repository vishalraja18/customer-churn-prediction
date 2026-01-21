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
