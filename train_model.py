import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

# Load dataset
df = pd.read_csv("loan_scoring.csv")

print("Dataset columns:")
print(df.columns)

# ===== FIX DATA FORMAT =====
df["monthly_income"] = df["monthly_income"].replace(",", "", regex=True).astype(float)
df["loan_amount"] = df["loan_amount"].replace(",", "", regex=True).astype(float)
df["monthly_expenses"] = df["monthly_expenses"].replace(",", "", regex=True).astype(float)

# ===== FEATURES =====
features = [
"age",
"monthly_income",
"loan_amount",
"credit_score",
"employment_years",
"credit_history_years"
]

X = df[features]

# Target
y = df["loan_status"]

# Train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

# ===== MODEL EVALUATION =====
y_pred = model.predict(X_test)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall:", recall_score(y_test,y_pred))
print("ROC-AUC:", roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))

# Save model
joblib.dump(model,"risk_model.pkl")

print("\nModel trained successfully!")
