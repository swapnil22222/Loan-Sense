import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Loan-Sense", page_icon="💳", layout="wide")

st.markdown("""
<style>
.main {
    padding-top: 20px;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #374151;
    margin-bottom: 15px;
}
.big-text {
    font-size: 28px;
    font-weight: bold;
}
.mid-text {
    font-size: 18px;
}
.approve {
    color: #16a34a;
    font-size: 34px;
    font-weight: bold;
}
.review {
    color: #d97706;
    font-size: 34px;
    font-weight: bold;
}
.reject {
    color: #dc2626;
    font-size: 34px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Loan-Sense")
st.caption("AI-powered lending decision dashboard")

st.sidebar.header("Applicant Details")

data = {}

data["age"] = st.sidebar.number_input("Age", 18, 100, 30)
data["gender"] = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
data["marital_status"] = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
data["education_level"] = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
data["annual_income"] = st.sidebar.number_input("Annual Income", 1000.0, 1000000.0, 50000.0)
data["monthly_income"] = st.sidebar.number_input("Monthly Income", 100.0, 100000.0, 4000.0)
data["employment_status"] = st.sidebar.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Student", "Retired"])
data["debt_to_income_ratio"] = st.sidebar.number_input("Debt To Income Ratio", 0.0, 1.0, 0.2)
data["credit_score"] = st.sidebar.number_input("Credit Score", 300, 900, 700)
data["loan_amount"] = st.sidebar.number_input("Loan Amount", 500.0, 1000000.0, 10000.0)
data["loan_purpose"] = st.sidebar.selectbox("Loan Purpose", ["Car", "Home", "Business", "Medical", "Education", "Vacation", "Other", "Debt consolidation"])
data["interest_rate"] = st.sidebar.number_input("Interest Rate", 1.0, 30.0, 10.0)
data["loan_term"] = st.sidebar.number_input("Loan Term", 12, 120, 36)
data["installment"] = st.sidebar.number_input("Installment", 50.0, 50000.0, 300.0)
data["grade_subgrade"] = st.sidebar.selectbox("Grade", ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5","C1","C2","C3","C4","C5","D1","D2","D3","D4","D5","E1","E2","E3","E4","E5","F1","F2","F3","F4","F5"])
data["num_of_open_accounts"] = st.sidebar.number_input("Open Accounts", 0, 50, 5)
data["total_credit_limit"] = st.sidebar.number_input("Total Credit Limit", 500.0, 1000000.0, 50000.0)
data["current_balance"] = st.sidebar.number_input("Current Balance", 0.0, 1000000.0, 10000.0)
data["delinquency_history"] = st.sidebar.number_input("Delinquency History", 0, 20, 0)
data["public_records"] = st.sidebar.number_input("Public Records", 0, 20, 0)
data["num_of_delinquencies"] = st.sidebar.number_input("Number Of Delinquencies", 0, 20, 0)

input_df = pd.DataFrame([data])

for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

if st.sidebar.button("Analyze Applicant"):

    prob = model.predict_proba(input_df)[0][1]

    score = int(300 + prob * 600)

    if data["debt_to_income_ratio"] > 0.4:
        score -= 35

    score -= data["num_of_delinquencies"] * 15

    score = max(300, min(900, score))

    if score >= 750:
        decision = "Approve"
        risk = "Low Risk"
        css = "approve"
    elif score >= 650:
        decision = "Review"
        risk = "Moderate Risk"
        css = "review"
    else:
        decision = "Reject"
        risk = "High Risk"
        css = "reject"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Repayment Probability", f"{round(prob*100,2)}%")

    with c2:
        st.metric("Risk Score", score)

    with c3:
        st.metric("Risk Level", risk)

    st.progress(score / 900)

    st.markdown(
        f"<div class='card'><div class='{css}'>{decision}</div><div class='mid-text'>Final Lending Decision</div></div>",
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:
        st.markdown("<div class='card'><div class='big-text'>Applicant Summary</div></div>", unsafe_allow_html=True)
        st.write("Income:", data["annual_income"])
        st.write("Credit Score:", data["credit_score"])
        st.write("Loan Amount:", data["loan_amount"])
        st.write("Employment:", data["employment_status"])

    with b:
        st.markdown("<div class='card'><div class='big-text'>Risk Factors</div></div>", unsafe_allow_html=True)
        st.write("DTI Ratio:", data["debt_to_income_ratio"])
        st.write("Delinquencies:", data["num_of_delinquencies"])
        st.write("Open Accounts:", data["num_of_open_accounts"])
        st.write("Current Balance:", data["current_balance"])

else:
    st.info("Fill details in sidebar and click Analyze Applicant")