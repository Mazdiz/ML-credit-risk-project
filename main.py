import streamlit as st
from predictionhelper import predict

st.title("Lauki Finance: Credit Risk Modelling")

st.write("Enter applicant details below:")

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Row 1
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, step = 1)
with col2:
    loan_tenure_months = st.number_input("Loan Tenure (months)", min_value=1, max_value=360)
with col3:
    number_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, max_value=50, step = 1)

# Row 2
with col1:
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0)
with col2:
    deliquency_ratio = st.number_input("Delinquency Ratio (%)", min_value=0.0, max_value=100.0)
with col3:
    average_dpd = st.number_input("Average Days Past Due", min_value=0, max_value=365)

# Row 3
with col1:
    residence_type = st.selectbox("Residence Type", ['Owned', 'Mortgage', 'Rented'])
with col2:
    loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Auto', 'Personal', 'Education'])
with col3:
    loan_type = st.selectbox("Loan Type", ['Secured', 'Unsecured'])

# Row 4: loan amount and income for internal calculation
with col1:
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
with col2:
    monthly_income = st.number_input("Monthly Income", min_value=0.0)
with col3:
    if monthly_income > 0:
        loan_to_income = loan_amount / monthly_income
    else:
        loan_to_income = 0.0

    st.markdown(f"Loan-to-Income Ratio:  \n\n{loan_to_income:.2f}")

# Submit button
if st.button("Calculate Risk"):
    probability, credit_score, rating = predict(age, loan_tenure_months, number_of_open_accounts, credit_utilization_ratio,
                                                deliquency_ratio, average_dpd, residence_type, loan_purpose, loan_type,
                                                loan_amount, monthly_income)

    st.write(f"Default Probability: {probability:.2%}")
    st.write(f"Credit Score: {credit_score}")
    st.write(f"Rating: {rating}")

