import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("loan_scoring.csv")

# Fix number format
df["monthly_income"] = df["monthly_income"].replace(",", "", regex=True).astype(float)
df["loan_amount"] = df["loan_amount"].replace(",", "", regex=True).astype(float)

# Encode gender
if "gender" in df.columns:
    df["gender"] = df["gender"].map({"male":1,"female":0})

# Features
features = [
    "age",
    "monthly_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "credit_history_years"
]

X = df[features]
y = df["loan_status"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X,y)

# Save model
joblib.dump(model,"risk_model.pkl")

print("Model trained successfully!")
