import streamlit as st
import joblib
import pandas as pd

# Load model
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

st.set_page_config(
    page_title="AI Loan Risk System",
    layout="wide"
)

st.title("🏦 AI Loan Risk Scoring Dashboard")
st.write("Hybrid AI + Rule Engine Loan Risk Evaluation")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age",18,70,30)

income = st.sidebar.number_input(
    "Annual Income ($)",
    min_value=10000,
    max_value=200000,
    value=50000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=1000,
    max_value=100000,
    value=20000
)

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    850,
    650
)

employment_years = st.sidebar.slider(
    "Employment Years",
    0,
    40,
    5
)

credit_history_years = st.sidebar.slider(
    "Credit History (Years)",
    0,
    30,
    5
)
existing_debt = st.sidebar.number_input(
    "Existing Debt ($)",
    min_value=0,
    max_value=100000,
    value=5000
)

late_payments = st.sidebar.slider(
    "Late Payments",
    0,
    20,
    0
)

loan_term = st.sidebar.selectbox(
    "Loan Term (months)",
    [12,24,36,48,60]
)
education = st.sidebar.selectbox(
    "Education",
    ["High School","Bachelor","Master","PhD"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["employed","self-employed","unemployed"]
)

# ---------------- RULE ENGINE ---------------- #

def rule_engine(age,income,loan_amount,credit_score):

    if credit_score < 450:
        return "Reject"

    if loan_amount > income * 0.9:
        return "Review"

    return "Pass"

# ---------------- PREDICTION ---------------- #

if st.sidebar.button("Evaluate Application"):

    monthly_income = income / 12
    monthly_expenses = monthly_income * 0.3
    
    data = pd.DataFrame({
    "age":[age],
    "gender":["male"],
    "employment_years":[employment_years],
    "employment_status":[employment_status],
    "monthly_income":[monthly_income],
    "monthly_expenses":[monthly_expenses],
    "credit_history_years":[credit_history_years],
    "past_default":[0],
    "residence_type":["rent"],
    "loan_amount":[loan_amount],
    "education":[education],
    "loan_intent":["personal"],
    "interest_rate":[12],
    "loan_percent_income":[loan_amount/monthly_income],
    "credit_score":[credit_score],
    "existing_debt":[existing_debt],
    "late_payments":[late_payments],
    "loan_term":[loan_term]
})
    rule_result = rule_engine(age,income,loan_amount,credit_score)

    data["gender"] = data["gender"].map({
    "male":1,
    "female":0
})

    data["employment_status"] = data["employment_status"].map({
    "employed":2,
    "self-employed":1,
    "unemployed":0
})

    data["education"] = data["education"].map({
    "High School":0,
    "Bachelor":1,
    "Master":2,
    "PhD":3
})

    data["residence_type"] = data["residence_type"].map({
    "rent":0,
    "own":1,
    "mortgage":2
})

    data["loan_intent"] = data["loan_intent"].map({
    "personal":0,
    "education":1,
    "medical":2,
    "venture":3,
    "home_improvement":4
})

    data = data[model.feature_names_in_]

    risk = model.predict_proba(data)[0][1]

    # ---------------- OUTPUT ---------------- #

    st.subheader("Risk Assessment")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Default Probability",f"{risk*100:.2f}%")

    with col2:

        if risk < 0.3:
            level = "Low Risk"
            color = "🟢"
        elif risk < 0.6:
            level = "Medium Risk"
            color = "🟡"
        else:
            level = "High Risk"
            color = "🔴"

        st.metric("Risk Level",f"{color} {level}")

    with col3:
        st.metric("Rule Engine",rule_result)

    st.divider()

    if rule_result == "Reject":
        st.error("Application Rejected by Rule Engine")

    elif risk > 0.65:
        st.warning("High risk detected. Manual review recommended.")

    else:
        st.success("Loan Application Approved")

    st.subheader("Customer Data")

    st.dataframe(data)
