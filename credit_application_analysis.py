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