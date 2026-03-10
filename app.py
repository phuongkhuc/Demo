import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Loan Risk System", layout="wide")

st.title("🏦 AI Loan Risk Scoring Dashboard")
st.write("Hybrid AI + Rule Engine Loan Risk Evaluation")

# ---------------- LOAD DATA ---------------- #

@st.cache_resource
def train_model():

    df = pd.read_csv("loan_scoring.csv")

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

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

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

# ---------------- RULE ENGINE ---------------- #

def rule_engine(credit_score, income, loan_amount):

    if credit_score < 450:
        return "Reject"

    if loan_amount > income * 0.9:
        return "Review"

    return "Pass"

# ---------------- PREDICTION ---------------- #

if st.sidebar.button("Evaluate Application"):

    monthly_income = income / 12

    data = pd.DataFrame({
        "age":[age],
        "monthly_income":[monthly_income],
        "loan_amount":[loan_amount],
        "credit_score":[credit_score],
        "employment_years":[employment_years],
        "credit_history_years":[credit_history_years]
    })

    risk = model.predict_proba(data)[0][1]

    rule_result = rule_engine(credit_score,income,loan_amount)

# ---------------- OUTPUT ---------------- #

    st.subheader("Risk Assessment")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Default Probability",f"{risk*100:.2f}%")

    with col2:

        if risk < 0.3:
            level="Low Risk"
            color="🟢"
        elif risk < 0.6:
            level="Medium Risk"
            color="🟡"
        else:
            level="High Risk"
            color="🔴"

        st.metric("Risk Level",f"{color} {level}")

    with col3:
        st.metric("Rule Engine",rule_result)

    if rule_result=="Reject":
        st.error("Application Rejected")

    elif risk>0.65:
        st.warning("Manual review recommended")

    else:
        st.success("Loan Approved")
