import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="AI Credit Decision System", layout="wide")

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

# ---------------- LOAD DATA ---------------- #

@st.cache_resource
def load_model():

    df = pd.read_csv("credit_data_processed.csv")
    internal_df = pd.read_csv("Internal_mock_data_20k.csv")
    cic_df = pd.read_csv("CIC_mock_data_100k.csv")

    internal_df["national_id"] = internal_df["national_id"].astype(str)
    cic_df["national_id"] = cic_df["national_id"].astype(str)

    # remove commas
    df = df.replace(",", "", regex=True)

    # convert everything numeric when possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # keep only numeric columns
    df = df.select_dtypes(include=["int64", "float64"])

    df = df.dropna()

# ---------------- TRAIN MODEL ---------------- #
    features = [
        "age",
        "monthly_income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "credit_history_years"
    ]
    credit_df = credit_df.dropna()
   
    X = df[features]
    y = df["loan_status"]

    model = RandomForestClassifier(n_estimators=100,random_state=42)

    model.fit(X, y)

    return model


# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Customer Information")
national_id = st.sidebar.text_input("National ID")

dob = st.sidebar.date_input(
    "Date of Birth"
)
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
    ["full time","part time","self employed","unemployed"]
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
    "Loan Intent",
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

existing_debt = st.sidebar.number_input(
    "Existing Monthly Debt ($)",
    min_value=0,
    max_value=10000,
    value=500
)

max_dpd = st.sidebar.slider(
    "Max DPD (Days Past Due)",
    0,120,0
)

# ---------------- RULE ENGINE ---------------- #

def calculate_age(dob):

    today = datetime.today()

    age = today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
    )

    return age

def knockout_rules(age,nationality):

    if age < 18 or age > 65:
        return "Reject"

    if nationality != "Vietnam":
        return "Reject"

    return "Pass"                      
                       
def detect_(national_id):

    if national_id in internal_df["national_id"].astype(str).values:
        return "ETB"
    else:
        return "NTB"
         
def customer_screening(, is_blacklisted, is_fraud):

    # ETB → check blacklist
    if  == "ETB":

        if is_blacklisted or is_fraud:
            return "Reject"

        return "Pass"

def get_cic_data(national_id):

    row = cic_df[cic_df["national_id"] == str(national_id)]

    if row.empty:
        return None

    return row.iloc[0]

def cic_rules(cic_row):

    if cic_row is None:
        return "Pass", None, None

    max_dpd = cic_row["max_dpd"]
    credit_score = cic_row["credit_score"]
    existing_debt = cic_row["existing_debt_obligations"]

    # Rule 1: DPD
    if max_dpd > 30:
        return "Reject", max_dpd, credit_score

    # Rule 2: Credit score
    if 0 < credit_score <= 430:
        return "Reject", max_dpd, credit_score

    return "Pass", max_dpd, credit_score

def capacity_rules(monthly_income, dti_1, risk):

    # Rule 1: minimum income
    if monthly_income < 500:
        return "Reject", "Income below minimum requirement"

    # Rule 2: DTI
    existing_debt = cic_row["existing_debt_obligations"]
    dti_1 = existing_debt / monthly_income
    
    if dti_1 >= 0.5:
        return "Reject", "Debt-to-income exceeds 50%"

    # Rule 3: ML risk
    if risk < 0.7:
        return "Reject", "ML risk score below threshold"

    return "Pass", "Capacity rules passed"
     


# ---------------- Decision Matrix (Approve / Partial / Manual) ---------------- #

def decision_matrix(,risk,credit_score,dti_2,
                    loan_amount,monthly_income,existing_debt):

    if  == "NTB":

        # ----- HIGH RISK SCORE (>=0.7) -----

        if 0.7 <= risk <= 1:

            if credit_score >= 570 and dti_2 <= 0.5:
                return "Approve", loan_amount

            if 431 <= credit_score < 570 and dti_2 <= 0.36:
                return "Approve", loan_amount

            if 431 <= credit_score < 570 and 0.36 < dti_2 <= 0.5:

                limit = (0.36*monthly_income-existing_debt)/0.05
                limit = int(limit // 1000 * 1000)
                return "Partial Approve", limit

            if pd.isna(credit_score) and dti_2 <= 0.36:
                return "Approve", loan_amount

            if pd.isna(credit_score) and 0.36 < dti_2 <= 0.5:

                limit = (0.36 * monthly_income - existing_debt)/0.05
                limit = int(limit // 1000 * 1000)
                return "Partial Approve", limit


        # ----- MEDIUM RISK (False positive zone) -----

        if 0.5 <= risk < 0.7:

            if credit_score >= 570 and dti_2 <= 0.36:
                return "Approve", loan_amount

            if credit_score >= 570 and 0.36 < dti_2 <= 0.5:
                limit = (0.36*monthly_income)/0.05
                limit = int(limit // 1000 * 1000)
                return "Partial Approve", limit

            if 431 <= credit_score < 570 and dti_2 <= 0.36:
                return "Manual Review",0

            if 431 <= credit_score < 570 and 0.36 < dti_2 <= 0.5:
                return "Reject",0

            if pd.isna(credit_score) and dti_2 <= 0.36:
                return "Manual Review", 0

            if pd.isna(credit_score) and 0.36 < dti_2 <= 0.5:
                return "Reject", 0
        
        return "Reject",0

    # ---------- ETB ----------
    if  == "ETB":

        if 0.7 <= risk <= 1 and dti_2 <= 0.5:

            if credit_score >= 431 or pd.isna(credit_score):
                return "Approve", loan_amount

        if 0.5 <= risk < 0.7 and dti_2 <= 0.5:

            if credit_score >= 431 or pd.isna(credit_score):
                return "Manual Review", 0

        return "Reject", 0

# ---------------- EVALUTE ---------------- #

if st.sidebar.button("Evaluate Application"):

    age = calculate_age(dob)  
    customer_type = detect_customer_type(national_id, internal_df)

    
    loan_percent_income = loan_amount / monthly_income

    expense_to_income = monthly_expenses / monthly_income

    dti_1 = existing_debt / monthly_income

    new_debt = loan_amount * 0.05

    dti_2 = (existing_debt + new_debt) / monthly_income

    
   
    

    st.write("Customer Type:", customer_type)
   
    # ---------------- CIC DATA ---------------- #

    cic_row = get_cic_data(national_id)

    cic_result, max_dpd, credit_score = cic_rules(cic_row)

    if cic_result == "Reject":

        st.error("Rejected by CIC rule")
        st.write("Max DPD:", max_dpd)
        st.write("Credit Score:", credit_score)

        decision = "Reject"
        limit = 0

        st.stop()
    
  # ----- ML prediction -----
    
    data = pd.DataFrame({

        "age":[age],
        "gender":[1 if gender=="Male" else 0],
        "employment_years":[employment_years],
        "employment_status":[
            {"full time":0,"part time":1,"self employed":2,"unemployed":3}[employment_status]
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
    existing_debt = cic_row["existing_debt_obligations"]
    
    st.write("Risk probability:", risk)

    new_debt = loan_amount * 0.05
    dti_2 = (existing_debt + new_debt) / monthly_income

    screening = customer_screening(
        customer_type,
        is_blacklisted,
        is_fraud,
        max_dpd
)

    if screening == "Reject":

        decision = "Reject"
        limit = 0

    else:
        rule_result = knockout_rules(
            age,
            nationality,
            is_blacklisted,
            is_fraud,
            max_dpd,
            credit_score,
            monthly_income,
            dti_1,
            risk
    )

    
    capacity_result, capacity_message = capacity_rules(
         monthly_income,
         dti_1,
         risk
)

    if capacity_result == "Reject":

        st.error(capacity_message)

        decision = "Reject"
        limit = 0

        st.stop()
    
    decision, limit = decision_matrix(
        customer_type,
        risk,
        credit_score,
        dti_2,
        loan_amount,
        monthly_income,
        existing_debt
)

    
    st.markdown('<div class="card">', unsafe_allow_html=True)
 

# ---------------- OUTPUT ---------------- #

    st.subheader("📊 CIC Data")

    cic_display = pd.DataFrame({
        "Credit Score":[credit_score],
        "Max DPD":[max_dpd],
        "Existing Debt":[existing_debt]
})

    st.dataframe(cic_display)

    st.subheader("Capacity Check")
    capacity_df = pd.DataFrame({
        "Monthly Income":[monthly_income],
        "Existing Debt":[existing_debt],
        "DTI":[round(dti_1,2)],
        "ML Risk":[round(risk,2)]
})

    st.dataframe(capacity_df)

    # Tách KPI dashboard section
    
    st.subheader("📊 AI Risk Assessment Dashboard")

     # ---------------- ROW 1: METRICS ---------------- #

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Default Probability",f"{risk*100:.2f}%",
                  help="Probability of customer default predicted by ML model"
        )

    with col2:

        if risk < 0.4:
            level = "LOW RISK"
            color = "🟢"
        elif risk < 0.7:
            level = "MEDIUM RISK"
            color = "🟡"
        else:
            level = "HIGH RISK"
            color = "🔴"

        st.metric("Risk Level",f"{color} {level}")

    with col3:
        st.metric("Rule Engine",rule_result)

    st.markdown('</div>', unsafe_allow_html=True)
    
#------
    
    st.markdown('<div class="card section-space">', unsafe_allow_html=True) 
    st.subheader("⚖️ Final Decision")
    st.write("Customer Type:", customer_type)

    st.write("Risk Score:", round(risk,3))

    st.write("DTI:", round(dti_2,2))

    st.write("Decision:", decision)
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
        
    st.markdown('</div>', unsafe_allow_html=True)
    
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
               'bar': {'color':"#1f2937"},
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
           title="Customer Financial Overview",
           color_discrete_map={
                "Income":"#3b82f6",
                "Expenses":"#f59e0b",
                "Loan":"#ef4444"
    }
)
        
       st.plotly_chart(fig2, use_container_width=True, key="finance_chart")

    st.markdown('</div>', unsafe_allow_html=True)
    
# ---------------- ROW 3 : RISK DISTRIBUTION ---------------- #
    #Tách Risk distribution

    st.markdown('<div class="card section-space">', unsafe_allow_html=True)


    #----

    st.subheader("📈Risk Distribution")
    risk_chart = pd.DataFrame({
       "Type":["Low Risk","High Risk"],
       "Probability":[1-risk,risk]
})

    fig3 = px.pie(
        risk_chart,
        names="Type",
        values="Probability",
        color="Type",
        color_discrete_map={
             "Low Risk":"#22c55e",
             "High Risk":"#ef4444"
    },
        title="Risk Probability Distribution"
)

    st.plotly_chart(fig3, use_container_width=True, key="risk_distribution")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ROW 4 : CUSTOMER DATA ---------- #

    st.subheader("📝 Customer Data")

    st.dataframe(data)   

