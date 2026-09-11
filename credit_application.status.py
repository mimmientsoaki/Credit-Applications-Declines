# ==========================================
# 🏦 CREDIT APPLICATIONS ANALYSIS
# ==========================================

# ==========================================
# 📥 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd

# ==========================================
# 📂 2. LOAD DATA
# ==========================================
df = pd.read_csv("credit_applications.csv")

# ==========================================
# 🔎 3. UNDERSTAND THE DATA
# ==========================================
print("\nFirst 5 Records",df.head())
print("\nDataset Shape",df.shape)
print("\nDescription of Dataset\n",df.describe())
df.info()

# ==========================================
# 🧹 4. DATA QUALITY & CLEANING
# ==========================================
print("\nMissing Values Count\n",df.isnull().sum().sort_values(ascending=False))
print("\nIndependent Variables\n",df["Application_Status"].isnull().sum())
print("Duplicate Records Count\n",df.duplicated().sum())

# ==========================================
# 📊 5. EXPLORATORY DATA ANALYSIS
# ==========================================


# ==========================================
# 📈 6. STATISTICAL ANALYSIS
# ==========================================


# ==========================================
# 🤖 7. MACHINE LEARNING
# ==========================================


# ==========================================
# 📏 8. MODEL EVALUATION
# ==========================================


# ==========================================
# ⭐ 9. FEATURE IMPORTANCE
# ==========================================


# ==========================================
# 💼 10. BUSINESS INSIGHTS
# ==========================================