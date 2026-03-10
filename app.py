import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Loan Risk System", layout="wide")

st.title("🏦 AI Loan Risk Scoring Dashboard")
st.write("Hybrid AI + Rule Engine Loan Risk Evaluation")

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():

    df = pd.read_csv("loan_scoring.csv")

    # remove commas
    df = df.replace(",", "", regex=True)

    # convert everything numeric when possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # keep only numeric columns
    df = df.select_dtypes(include=["int64", "float64"])

    df = df.dropna()

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

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


model = load_model()

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age",18,70,30)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

employment_years = st.sidebar.slider(
    "Employment Years",
    0,40,5
)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["employed","self-employed","unemployed"]
)

monthly_income = st.sidebar.number_input(
    "Monthly Income ($)",
    min_value=500,
    max_value=20000,
    value=4000
)

monthly_expenses = st.sidebar.number_input(
    "Monthly Expenses ($)",
    min_value=0,
    max_value=15000,
    value=1500
)

credit_history_years = st.sidebar.slider(
    "Credit History (Years)",
    0,30,5
)

past_default = st.sidebar.selectbox(
    "Past Default",
    ["No","Yes"]
)

residence_type = st.sidebar.selectbox(
    "Residence Type",
    ["rent","own","mortgage"]
)

loan_amount = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=1000,
    max_value=100000,
    value=20000
)

education = st.sidebar.selectbox(
    "Education",
    ["High School","Bachelor","Master","PhD"]
)

loan_intent = st.sidebar.selectbox(
    "Loan Purpose",
    ["personal","education","medical","venture","home_improvement"]
)

interest_rate = st.sidebar.slider(
    "Interest Rate (%)",
    5,25,12
)

credit_score = st.sidebar.slider(
    "Credit Score",
    300,850,650
)

nationality = st.sidebar.selectbox(
    "Nationality",
    ["Vietnam","Other"]
)

customer_type = st.sidebar.selectbox(
    "Customer Type",
    ["NTB","ETB"]
)

existing_debt = st.sidebar.number_input(
    "Existing Monthly Debt ($)",
    min_value=0,
    max_value=10000,
    value=500
)

expected_credit_limit = st.sidebar.number_input(
    "Expected Credit Limit ($)",
    min_value=1000,
    max_value=50000,
    value=10000
)

max_dpd = st.sidebar.slider(
    "Max DPD (Days Past Due)",
    0,120,0
)

is_blacklisted = st.sidebar.checkbox("Blacklisted")

is_fraud = st.sidebar.checkbox("Fraud List")

# ---------------- RULE ENGINE ---------------- #


def knockout_rules(age,nationality,is_blacklisted,is_fraud,max_dpd,
                   credit_score,monthly_income,dti,risk):

    if age < 18 or age > 65:
        return "Reject"

    if nationality != "Vietnam":
        return "Reject"

    if is_blacklisted or is_fraud:
        return "Reject"

    if max_dpd > 30:
        return "Reject"

    if credit_score <= 430:
        return "Reject"

    if monthly_income < 500:
        return "Reject"

    if dti >= 0.5:
        return "Reject"

    if risk < 0.7:
        return "Reject"

    return "Pass"

# ---------------- Decision Matrix (Approve / Partial / Manual) ---------------- #

def decision_matrix(customer_type,risk,credit_score,dti,
                    expected_credit_limit,monthly_income,existing_debt):

    if customer_type == "NTB":

        if risk >= 0.7 and credit_score >= 570 and dti <= 0.36:
            return "Approve", expected_credit_limit

        if risk >= 0.7 and 431 <= credit_score < 570 and dti <= 0.36:
            return "Approve", expected_credit_limit

        if risk >= 0.7 and 431 <= credit_score < 570 and 0.36 < dti <= 0.5:

            limit = (0.36*monthly_income-existing_debt)/0.05
            return "Partial Approve", int(limit)

        if 0.5 <= risk < 0.7:
            return "Manual Review",0

        return "Reject",0

    if customer_type == "ETB":

        if risk >= 0.7 and credit_score >= 431 and dti <= 0.5:
            return "Approve", expected_credit_limit

        if 0.5 <= risk < 0.7:
            return "Manual Review",0

        return "Reject",0

# ---------------- PREDICTION ---------------- #

if st.sidebar.button("Evaluate Application"):

    loan_percent_income = loan_amount / (monthly_income * 12)

    expense_to_income = monthly_expenses / monthly_income

    new_debt = expected_credit_limit * 0.05

    dti = (existing_debt + new_debt) / monthly_income
   
    data = pd.DataFrame({

        "age":[age],
        "gender":[1 if gender=="Male" else 0],
        "employment_years":[employment_years],
        "employment_status":[
            {"unemployed":0,"self-employed":1,"employed":2}[employment_status]
        ],
        "monthly_income":[monthly_income],
        "monthly_expenses":[monthly_expenses],
        "credit_history_years":[credit_history_years],
        "past_default":[1 if past_default=="Yes" else 0],
        "residence_type":[
            {"rent":0,"own":1,"mortgage":2}[residence_type]
        ],
        "loan_amount":[loan_amount],
        "education":[
            {"High School":0,"Bachelor":1,"Master":2,"PhD":3}[education]
        ],
        "loan_intent":[
            {"personal":0,"education":1,"medical":2,"venture":3,"home_improvement":4}[loan_intent]
        ],
        "interest_rate":[interest_rate],
        "loan_percent_income":[loan_percent_income],
        "credit_score":[credit_score],
        "expense_to_income":[expense_to_income]

    })



    # prediction

    risk = model.predict_proba(data)[0][1]

    st.write("Risk probability:", risk)
    

    new_debt = expected_credit_limit * 0.05
    dti = (existing_debt + new_debt) / monthly_income

    rule_result = knockout_rules(
        age,
        nationality,
        is_blacklisted,
        is_fraud,
        max_dpd,credit_score,
        monthly_income,dti,risk
)

    if rule_result == "Reject":

       decision = "Reject"
       limit = 0

    else:

       decision,limit = decision_matrix(
         customer_type,
         risk,
         credit_score,
         dti,
         expected_credit_limit,
         monthly_income,
         existing_debt
    )

    st.subheader("Final Decision")

    st.metric("Decision",decision)

    if decision == "Approve":
       st.success(f"Approved Limit: ${limit}")

    elif decision == "Partial Approve":
       st.warning(f"Adjusted Limit: ${limit}")

    elif decision == "Manual Review":
       st.info("Application requires manual review")

    else:
       st.error("Application Rejected")


    # ----- Gauge ----- #

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        title={'text': "Default Risk (%)"},
        gauge={
            'axis': {'range':[0,100]},
            'steps':[
                {'range':[0,30],'color':"lightgreen"},
                {'range':[30,60],'color':"yellow"},
                {'range':[60,100],'color':"salmon"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True, key="risk_gauge")

   # ---------------- Risk Distribution Chart ---------------- #

    risk_chart = pd.DataFrame({
       "Type":["Low Risk","Medium Risk","High Risk"],
       "Probability":[
             max(0,1-risk-0.3),
             min(risk,0.6),
             risk
    ]
})

    fig3 = px.pie(
        risk_chart,
        names="Type",
        values="Probability",
        title="Risk Distribution"
)

    st.plotly_chart(fig3, use_container_width=True, key="risk_distribution")

    # ---------------- OUTPUT ---------------- #

    risk = model.predict_proba(data)[0][1]

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



    # ---------------- financial chart ---------------- #

    finance_data = pd.DataFrame({
        "Category":["Income","Expenses","Loan"],
        "Amount":[monthly_income, monthly_expenses, loan_amount]
})

    fig2 = px.bar(
       finance_data,
       x="Category",
       y="Amount",
       color="Category",
       title="Customer Financial Overview"
)

    st.plotly_chart(fig2, use_container_width=True, key="finance_chart")

 

    # ---------------- Layout fintech dashboard ---------------- #

    st.subheader("Risk Assessment Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    st.plotly_chart(fig3, use_container_width=True)
