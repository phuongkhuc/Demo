import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Loan Risk System", layout="wide")

st.markdown("""
<style>

/* page padding */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

/* card style */
.card{
    background-color:white;
    padding:25px;
    border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
    margin-bottom:25px;
}

/* section title */
.section-title{
    font-size:20px;
    font-weight:600;
    margin-bottom:10px;
}

/* spacing */
.section-space{
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

st.title("AI Credit Risk Decision System")
st.caption("Hybrid Machine Learning + Rule Engine")

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():

    df = pd.read_csv("credit_data_processed.csv")

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
    ["full_time","part_time","self_employed","unemployed"]
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
    ["mortgage","other","own","rent"]
)

loan_amount = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=1000,
    max_value=100000,
    value=20000
)

education = st.sidebar.selectbox(
    "Education",
    ["associate","bachelor","doctorate","high school","master"]
)

loan_intent = st.sidebar.selectbox(
    "Loan Purpose",
    ["debt consolidation","education","home improvement","medical","personal","venture"]
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

        # ----- HIGH RISK SCORE (>=0.7) -----

        if risk >= 0.7:

            if credit_score >= 570 and dti <= 0.5:
                return "Approve", expected_credit_limit

            if 431 <= credit_score < 570 and dti <= 0.36:
                return "Approve", expected_credit_limit

            if 431 <= credit_score < 570 and 0.36 < dti <= 0.5:

                limit = (0.36*monthly_income-existing_debt)/0.05
                return "Partial Approve", int(limit)

            if pd.isna(credit_score) and dti <= 0.36:
                return "Approve", expected_credit_limit

            if pd.isna(credit_score) and 0.36 < dti <= 0.5:

                limit = (0.36*monthly_income-existing_debt)/0.05
                return "Partial Approve", int(limit)


        # ----- MEDIUM RISK (False positive zone) -----

        if 0.5 <= risk < 0.7:

            if credit_score >= 570 and dti <= 0.36:
                return "Approve", expected_credit_limit

            if credit_score >= 570 and 0.36 < dti <= 0.5:

                limit = (0.36*monthly_income-existing_debt)/0.05
                return "Partial Approve", int(limit)

            if 431 <= credit_score < 570 and dti <= 0.36:
                return "Manual Review",0

            if 431 <= credit_score < 570 and 0.36 < dti <= 0.5:
                return "Reject",0

        return "Reject",0


    # ---------- ETB ----------

    if customer_type == "ETB":

        if risk >= 0.7 and credit_score >= 431 and dti <= 0.5:
            return "Approve", expected_credit_limit

        if 0.5 <= risk < 0.7 and credit_score >= 431 and dti <= 0.5:
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
            {"full_time":0,"part_time":1,"self_employed":2,"unemployed":3}[employment_status]
        ],
        "monthly_income":[monthly_income],
        "monthly_expenses":[monthly_expenses],
        "credit_history_years":[credit_history_years],
        "past_default":[1 if past_default=="Yes" else 0],
        "residence_type":[
            {"mortgage":0,"other":1,"own":2,"rent":3}[residence_type]
        ],
        "loan_amount":[loan_amount],
        "education":[
            {"associate":0,"bachelor":1,"doctorate":2,"high school":3,"master":4}[education]
        ],
        "loan_intent":[
            {"debt consolidation":0,"education":1,"home improvement":2,"medical":3,"personal":4,"venture":5}[loan_intent]
        ],
        "interest_rate":[interest_rate],
        "loan_percent_income":[loan_percent_income],
        "credit_score":[credit_score],
        "expense_to_income":[expense_to_income]

    })



    # Knock-out rules, Decision matrix #
    
    data = data[model.feature_names_in_]
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
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Final Decision</div>', unsafe_allow_html=True)

    st.metric("Decision",decision)

    if decision == "Approve":
       st.success(f"Approved Limit: ${limit}")

    elif decision == "Partial Approve":
       st.warning(f"Adjusted Limit: ${limit}")

    elif decision == "Manual Review":
       st.info("Application requires manual review")

    else:
       st.error("Application Rejected")

# False Positive Tuning Zone

    if 0.5 <= risk < 0.7:
        st.info("⚠ Medium Risk Zone → Sent to Manual Review to reduce False Positives")
    
    col4 = st.metric("AI Confidence Zone",
                 "Manual Review" if 0.5<=risk<0.7 else "Auto Decision")

# ---Alert-- #
    
    if rule_result == "Reject":
        st.error("Application rejected by rule engine")

    elif 0.5 <= risk < 0.7:
        st.warning("Medium risk detected → Manual review required")

    elif risk >= 0.7:
        st.success("Low default risk → Auto approval possible")
 

# ---------------- OUTPUT ---------------- #

    # Tách KPI dashboard section

    st.markdown('<div class="card section-space">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">AI Risk Assessment Dashboard</div>', unsafe_allow_html=True)

    st.subheader("📊 AI Risk Assessment Dashboard")

     # ---------------- ROW 1: METRICS ---------------- #

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Default Probability",f"{risk*100:.2f}%",
                  help="Probability of customer default predicted by ML model"
        )

    with col2:

        if risk < 0.3:
            level = "LOW RISK"
            color = "🟢"
        elif risk < 0.6:
            level = "MEDIUM RISK"
            color = "🟡"
        else:
            level = "HIGH RISK"
            color = "🔴"

        st.metric("Risk Level",f"{color} {level}")

    with col3:
        st.metric("Rule Engine",rule_result)

    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

    
   # ---------- ROW 2 : MAIN CHARTS ---------- #
    #Tách Chart section

    st.markdown('<div class="card section-space">', unsafe_allow_html=True)

    #---
   
    st.subheader("💳 Financial Overview")
    
    col1, col2 = st.columns(2)

    with col1:

        fig = go.Figure(go.Indicator(
           mode="gauge+number",
           value=risk * 100,
           title={'text': "Default Risk (%)"},
           gauge={
               'axis': {'range':[0,100]},
               'bar': {'color':"#111827"},
               'steps':[
                   {'range':[0,30],'color':"#22c55e"},
                   {'range':[30,60],'color':"#facc15"},
                   {'range':[60,100],'color':"#ef4444"}
            ]
        }
    ))

        st.plotly_chart(fig, use_container_width=True, key="risk_gauge")

        # ---------------- financial chart ---------------- #

    with col2:
       
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

    st.markdown('</div>', unsafe_allow_html=True)
    
# ---------------- ROW 3 : RISK DISTRIBUTION ---------------- #
    #Tách Risk distribution

    st.markdown('<div class="card section-space">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)


    #----

    st.subheader("📈Risk Distribution")
    risk_chart = pd.DataFrame({
       "Type":["Low Risk","High Risk"],
       "Probability":[risk,1-risk]
})

    fig3 = px.pie(
        risk_chart,
        names="Type",
        values="Probability",
        color="Type",
        color_discrete_map={
             "High Risk":"#ef4444",
             "Low Risk":"#22c55e"
    },
        title="Risk Probability Distribution"
)

    st.plotly_chart(fig3, use_container_width=True, key="risk_distribution")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ROW 4 : CUSTOMER DATA ---------- #

    st.subheader("Customer Data")

    st.dataframe(data)   

