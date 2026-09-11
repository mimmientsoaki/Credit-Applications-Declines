# ==========================================
# 📂 1. DATA UNDERSTANDING
# ==========================================

import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

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

pd.set_option("display.max_columns", None)
print("\nFeature Relationships")
numerical_features = [
    "Credit_Score",
    "Previous_Defaults",
    "Debt_to_Income",
    "Monthly_Income",
    "Existing_Debt",
    "Requested_Amount",
    "Employment_Years",
    "Previous_Applications",
    "Loan_Term_Months",
    "Age"
]
summary_table = df.groupby("Application_Status")[numerical_features].mean().round(2)
print(summary_table.to_string())


print("\nCategorical feature analysis summary")
print(pd.crosstab(df["Employment_Type"], df["Application_Status"], normalize="index") * 100)
print(pd.crosstab(df["Residential_Status"], df["Application_Status"], normalize="index") * 100)
print(pd.crosstab(df["Region"], df["Application_Status"], normalize="index") * 100)

# 📊 Residential Status vs Application Status
table = pd.crosstab(
    df["Residential_Status"],
    df["Application_Status"]
)
chi2, p, dof, expected = chi2_contingency(table)
print("Residential_Status")
print("Chi-Square:", chi2)
print("p-value:", p)

table2 = pd.crosstab(
    df["Employment_Type"],
    df["Application_Status"]
)
chi2, p, dof, expected = chi2_contingency(table2)
print("Employment_Type")
print("Chi-Square:", chi2)
print("p-value:", p)

table3 = pd.crosstab(
    df["Region"],
    df["Application_Status"]
)
chi2, p, dof, expected = chi2_contingency(table3)
print("Region")
print("Chi-Square:", chi2)
print("p-value:", p)

# ==========================================
# 📊 NUMERICAL FEATURES STATISTICAL ANALYSIS
# ==========================================
numerical_features = [
    "Credit_Score",
    "Previous_Defaults",
    "Debt_to_Income",
    "Monthly_Income",
    "Existing_Debt",
    "Requested_Amount",
    "Employment_Years",
    "Previous_Applications",
    "Loan_Term_Months",
    "Age"
]

approved = df[df["Application_Status"] == "Approved"]
declined = df[df["Application_Status"] == "Declined"]

results = []

for feature in numerical_features:
    t_stat, p_value = ttest_ind(
        approved[feature],
        declined[feature],
        equal_var=False,
        nan_policy="omit"
    )
    
    results.append({
        "Feature": feature,
        "T-Statistic": round(t_stat, 2),
        "P-Value": p_value
    })

results_df = pd.DataFrame(results)

print("=========================================")
print("\n📊 Numerical Features Statistical Analysis\n")
print(results_df.to_string(index=False))


# 🎯 Define Features and Target

model_df = df[df["Application_Status"].isin(["Approved", "Declined"])].copy()

categorical_features = ["Employment_Type", "Residential_Status", "Region"]

for feature in numerical_features:
    model_df[feature] = model_df[feature].fillna(model_df[feature].median())

for feature in categorical_features:
    model_df[feature] = model_df[feature].fillna(model_df[feature].mode().iloc[0])

X = model_df.drop(["Application_ID", "Application_Date", "Application_Status"], axis=1)
y = model_df["Application_Status"]

print("Features:", X.columns.tolist())
print("Target:", y.name)

# 🔤 Encode Categorical Variables

X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True
)

print(X.head())
print(X.dtypes)

# 🎯 Encode Target Variable


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

#Training Models
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42,stratify=y_encoded)

# ==========================================
# 📏 SCALE NUMERICAL FEATURES
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 1. Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Predictions:", y_pred[:10])

# ==========================================
# 📏 MODEL EVALUATION
# ==========================================
print("\n📊 Logistic Regression")
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", round(accuracy, 4))
# Precision, Recall and F1-score
print("\n📊 Classification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))
# Confusion Matrix
print("\n🔍 Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 🌳 2. DECISION TREE
# ==========================================
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

# ==========================================
# 📏 DECISION TREE EVALUATION
# ==========================================
print("\nDecision Tree")
print("\n🎯 Accuracy:", round(accuracy_score(y_test, tree_pred), 4))
print("\n📊 Classification Report")
print(classification_report(
    y_test,
    tree_pred,
    target_names=label_encoder.classes_
))
print("\n🔍 Confusion Matrix")
print(confusion_matrix(y_test, tree_pred))

# ==========================================
# 🌲 3. RANDOM FOREST
# ==========================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ==========================================
# 📏 RANDOM FOREST EVALUATION
# ==========================================

print("\nRandom Forest")
print("\n🎯 Accuracy:", round(accuracy_score(y_test, rf_pred), 4))
print("\n📊 Classification Report")
print(classification_report(
    y_test,
    rf_pred,
    target_names=label_encoder.classes_
))
print("\n🔍 Confusion Matrix")
print(confusion_matrix(y_test, rf_pred))

# ==========================================
# 🚀 4. GRADIENT BOOSTING
# ==========================================
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# ==========================================
# 📏 GRADIENT BOOSTING EVALUATION
# ==========================================
print("\n📊 Gradient Boosting")
print("\n🎯 Accuracy:", round(accuracy_score(y_test, gb_pred), 4))
print("\n📊 Classification Report")
print(classification_report(
    y_test,
    gb_pred,
    target_names=label_encoder.classes_
))
print("\n🔍 Confusion Matrix")
print(confusion_matrix(y_test, gb_pred))