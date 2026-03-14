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

credit_df = pd.read_csv("credit_data_processed.csv")

internal_df = pd.read_csv("Internal_mock_data_20k.csv")
internal_df["national_id"] = internal_df["national_id"].astype(str)

cic_df = pd.read_csv("CIC_mock_data_100k.csv")
cic_df["national_id"] = cic_df["national_id"].astype(str)

@st.cache_resource
def load_model():

  df = credit_df.copy()

  # remove commas
  df = df.replace(",", "", regex=True)


  df = df.dropna()

# ---------------- TRAIN MODEL ---------------- #
  features = [
        "age",
        "monthly_income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "credit_history_years",
        "loan_percent_income"
    ]
  df = df.dropna()
   
  X = df[features]
  y = df["loan_status"]

  model = RandomForestClassifier(n_estimators=200,random_state=42)

  model.fit(X, y)

  return model

model = load_model()

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Customer Information")
national_id = st.sidebar.text_input("National ID")

from datetime import date
dob = st.sidebar.date_input(
    "Date of Birth",
    value=date(1995,1,1),
    min_value=date(1950,1,1),
    max_value=date.today()
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

)

monthly_expenses = st.sidebar.number_input(
    "Monthly Expenses ($)",

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

existing_debt_obligations = st.sidebar.number_input(
    "Existing Monthly Debt ($)",

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
        return "Reject", "Age not eligible"

    if nationality != "Vietnam":
        return "Reject", "Nationality not supported"

    return "Pass", "Basic eligibility passed"                      
                       
def detect_customer_type(national_id):

    row = internal_df[internal_df["national_id"] == str(national_id)]

    if row.empty:
        return "NTB", None

    return "ETB", row.iloc[0]
  
def check_blacklist(customer_type, customer_row):

    if customer_type == "ETB" and customer_row is not None:

        if customer_row["is_blacklisted"] == 1:
            return "Reject", "Customer is blacklisted"

    return "Pass", "Blacklist check passed"

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
    existing_debt_obligations = cic_row["existing_debt_obligations"]

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

    dti_1 = existing_debt_obligations / monthly_income
    
    if dti_1 >= 0.5:
        return "Reject", "Debt-to-income exceeds 50%"

    # Rule 3: ML risk
    if risk < 0.7:
        return "Reject", "ML risk score below threshold"

    return "Pass", "Capacity rules passed"
     


# ---------------- Decision Matrix (Approve / Partial / Manual) ---------------- #

def decision_matrix(customer_type,risk,credit_score,dti_2,
                    loan_amount,monthly_income,existing_debt_obligations):

    if customer_type == "NTB":

        # ----- HIGH RISK SCORE (>=0.7) -----

        if 0.7 <= risk <= 1:

            if credit_score >= 570 and dti_2 <= 0.5:
                return "Approve", loan_amount

            if 431 <= credit_score < 570 and dti_2 <= 0.36:
                return "Approve", loan_amount

            if 431 <= credit_score < 570 and 0.36 < dti_2 <= 0.5:

                limit = (0.36*monthly_income-existing_debt_obligations)/0.05
                limit = int(limit // 1000 * 1000)
                return "Partial Approve", limit

            if pd.isna(credit_score) and dti_2 <= 0.36:
                return "Approve", loan_amount

            if pd.isna(credit_score) and 0.36 < dti_2 <= 0.5:

                limit = (0.36 * monthly_income - existing_debt_obligations)/0.05
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
                        
    if customer_type == "ETB":

        if 0.7 <= risk <= 1 and dti_2 <= 0.5:

            if credit_score >= 431 or pd.isna(credit_score):
                return "Approve", loan_amount

        if 0.5 <= risk < 0.7 and dti_2 <= 0.5:

            if credit_score >= 431 or pd.isna(credit_score):
                return "Manual Review", 0

        return "Reject", 0

# ---------------- EVALUTE ---------------- #
if monthly_income == 0:
    st.error("Monthly income cannot be zero")
    st.stop()
if st.sidebar.button("Evaluate Application"):

    loan_percent_income = loan_amount / monthly_income

    expense_to_income = monthly_expenses / monthly_income

    dti_1 = existing_debt_obligations / monthly_income

    new_debt = loan_amount * 0.05

    dti_2 = (existing_debt_obligations + new_debt) / monthly_income

    age = calculate_age(dob)
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
        "credit_score":[credit_score],
        "loan_percent_income":[loan_percent_income]

    })
    
    

    # Knock-out rules, Decision matrix #


    age = calculate_age(dob)

    # Age + nationality
    rule_result, message = knockout_rules(age, nationality)

    if rule_result == "Reject":
        st.error(message)
        st.stop()

    # Detect ETB / NTB
    customer_type, customer_row = detect_customer_type(national_id)

    st.write("Customer Type:", customer_type)

    #  check
    rule_result, message = check_blacklist(customer_type, customer_row)

    if rule_result == "Reject":
        st.error(message)
        st.stop()

    
    data = data[model.feature_names_in_]
    default_prob = model.predict_proba(data)[0][1]

    risk = 1 - default_prob
    st.write("Risk probability:", risk)
    
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
        existing_debt_obligations
)

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
 

# ---------------- OUTPUT ---------------- #

    st.markdown("""
    <style>

    .stApp{
       background:#f5f7fb;
}

    /* dashboard card */
    .card{
       background:white;
       border-radius:12px;
       padding:25px;
       margin-top:15px;
       margin-bottom:20px;
       border:1px solid #e6e8ef;
       box-shadow:0 2px 6px rgba(0,0,0,0.05);
}

    /* KPI card */
    .kpi{
       background:white;
       border-radius:10px;
       padding:18px;
       border:1px solid #e6e8ef;
       box-shadow:0 2px 6px rgba(0,0,0,0.05);
}

    </style>
    """, unsafe_allow_html=True)


    
    #Customer Summary Card
    with st.container():
    
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("👤 Customer Summary")

        summary_df = pd.DataFrame({
           "Customer Type":[customer_type],
           "Income":[monthly_income],
           "Loan Amount":[loan_amount],
           "Credit Score":[credit_score]
})

        st.dataframe(summary_df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

   
    #AI Risk Assessment Card

    st.markdown('<div class="card section-space">', unsafe_allow_html=True)

    st.subheader("🤖 AI Risk Assessment")

    col1,col2,col3 = st.columns(3)

    with col1:
       with st.container():
         st.markdown('<div class="kpi">', unsafe_allow_html=True)
         st.metric(
             "Approval Probability",
             f"{risk*100:.2f}%",
             help="Probability that the customer is a good borrower predicted by the ML model"
    )

    with col2:
        if risk < 0.5:
             level = "🔴 High Default Risk"
        elif risk < 0.7:
             level = "🟡 Medium Risk"
        else:
             level = "🟢 Low Default Risk"
            
      with st.container():   
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Risk Level",level)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
      with st.container():  
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Rule Engine",rule_result)
        st.markdown('</div>', unsafe_allow_html=True)

    
    #Financial Capacity Card
    col1,col2,col3 = st.columns(3)
    with col1:
      with st.container():  
         st.markdown('<div class="card">', unsafe_allow_html=True)

         st.subheader("💰 Financial Capacity")

         capacity_df = pd.DataFrame({
             "Monthly Income":[monthly_income],
             "Existing Debt":[existing_debt_obligations],
             "DTI":[round(dti_2,2)],
             "Loan Amount":[loan_amount]
})

         st.dataframe(capacity_df, use_container_width=True)
         st.markdown('</div>', unsafe_allow_html=True)

   
   #CIC Data Card
    with col2:
      with st.container():
         st.markdown('<div class="card">', unsafe_allow_html=True)
         st.subheader("🏦 CIC Bureau")

         cic_display = pd.DataFrame({
            "Credit Score":[credit_score],
            "Max DPD":[max_dpd],
            "Existing Debt":[existing_debt_obligations]
})

         st.dataframe(cic_display, use_container_width=True)
    
    with col3:
      with st.container():  
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Capacity Check")
        capacity_df = pd.DataFrame({
            "Monthly Income":[monthly_income],
            "Existing Debt":[existing_debt_obligations],
            "DTI":[round(dti_1,2)],
            "ML Risk":[round(risk,2)]
})

        st.dataframe(capacity_df)
        st.markdown('</div>', unsafe_allow_html=True)

    
    #Final Decision Card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True) 
        st.subheader("⚖️ Final Decision")
        st.write("Customer Type:", customer_type)

        st.write("Risk Score:", round(risk,3))

        st.write("DTI:", round(dti_2,2))

        st.write("Decision:", decision)
        if decision == "Approve":
           st.success(f"✅ Approved Limit: ${limit}")

        elif decision == "Partial Approve":
           st.warning(f"⚠ Adjusted Limit: ${limit}")

        elif decision == "Manual Review":
           st.info("Application requires manual review")

        else:
           st.error("Application Rejected")

# ---------- AI Confidence Zone ----------

        if risk < 0.5:
           confidence = "Low Confidence"
        elif risk < 0.7:
           confidence = "Manual Review Zone"
        else:
           confidence = "High Confidence"

        st.metric("AI Confidence Zone", confidence)
         
        st.markdown('</div>', unsafe_allow_html=True)

  #Chart



  

