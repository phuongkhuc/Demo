import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Loan Risk System", layout="wide")

st.title("🏦 AI Loan Risk Scoring Dashboard")
st.write("Hybrid AI + Rule Engine Loan Risk Evaluation")

# -------- LOAD MODEL -------- #

@st.cache_resource
def load_model():

    df = pd.read_csv("loan_scoring.csv")

    # encode categorical columns
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, encoders

model, encoders = load_model()

# -------- SIDEBAR INPUT -------- #

st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age",18,70,30)
income = st.sidebar.number_input("Annual Income ($)",10000,200000,50000)
loan_amount = st.sidebar.number_input("Loan Amount ($)",1000,100000,20000)
credit_score = st.sidebar.slider("Credit Score",300,850,650)

employment_years = st.sidebar.slider("Employment Years",0,40,5)
credit_history_years = st.sidebar.slider("Credit History Years",0,30,5)

existing_debt = st.sidebar.number_input("Existing Debt",0,100000,5000)
late_payments = st.sidebar.slider("Late Payments",0,20,0)

loan_term = st.sidebar.selectbox("Loan Term",[12,24,36,48,60])
education = st.sidebar.selectbox("Education",["High School","Bachelor","Master","PhD"])
employment_status = st.sidebar.selectbox("Employment Status",["employed","self-employed","unemployed"])

# -------- RULE ENGINE -------- #

def rule_engine(credit_score, income, loan_amount):

    if credit_score < 450:
        return "Reject"

    if loan_amount > income*0.9:
        return "Review"

    return "Pass"

# -------- PREDICTION -------- #

if st.sidebar.button("Evaluate Application"):

    monthly_income = income/12
    monthly_expenses = monthly_income*0.3

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
        "loan_status":[0]
    })

    # encode input
    for col in encoders:
        if col in data.columns:
            data[col] = encoders[col].transform(data[col])

    data = data.drop("loan_status", axis=1)

    risk = model.predict_proba(data)[0][1]

# -------- OUTPUT -------- #

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
        rule_result = rule_engine(credit_score,income,loan_amount)
        st.metric("Rule Engine",rule_result)

    if rule_result=="Reject":
        st.error("Application rejected by rule engine")

    elif risk>0.65:
        st.warning("High risk detected. Manual review recommended.")

    else:
        st.success("Loan Approved")
