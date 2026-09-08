# ==========================================
# 📂 1. DATA UNDERSTANDING
# ==========================================

import pandas as pd

df = pd.read_csv("credit_applications.csv")

print("\nFirst 5 Records",df.head())
print("\nDataset Shape",df.shape)
print("\nColumn Names",df.columns.tolist())
print("\n🔍 Dataset Information")
df.info()

print("\nMissing Values Count\n",df.isnull().sum().sort_values(ascending=False))
print("\nIndependent Variables\n",df["Application_Status"].isnull().sum())
print("\nDuplicate Records Count\n",df.duplicated().sum())

print("\nUnexpected values")
print(df["Application_Status"].value_counts())
print(df["Employment_Type"].value_counts())
print(df["Residential_Status"].value_counts())
print(df["Region"].value_counts())

print("\n Numerical Variables Summary\n")
print(df.describe())

print("\nFeature Relationships\n")
print(df.groupby("Application_Status")["Credit_Score"].mean())
print(df.groupby("Application_Status")["Previous_Defaults"].mean())
print(df.groupby("Application_Status")["Debt_to_Income"].mean())
print(df.groupby("Application_Status")["Monthly_Income"].mean())
print(df.groupby("Application_Status")["Existing_Debt"].mean())
print(df.groupby("Application_Status")["Requested_Amount"].mean())
print(df.groupby("Application_Status")["Employment_Years"].mean())
print(df.groupby("Application_Status")["Previous_Applications"].mean())
print(df.groupby("Application_Status")["Loan_Term_Months"].mean())
print(df.groupby("Application_Status")["Age"].mean())