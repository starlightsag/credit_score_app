# app.py

import streamlit as st
import numpy as np
import joblib
from PIL import Image


# Load and display logo
logo = Image.open("logo.png")  # Ensure you have a logo.png in the same directory
st.image(logo, width=100)  # Adjust width as needed

# Load model and scaler
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")


st.title("Credit Risk Prediction System")


st.write("Enter the customer details below:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Monthly Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_score = st.select_slider(
    "Credit Score",
    options=list(range(300, 901, 10)),
    value=600,
    format_func=lambda x: f"{x} points"
)

employment_years = st.number_input("Years of Employment", min_value=0, max_value=50)
education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
existing_loans_count = st.number_input("Existing Loans Count", min_value=0, max_value=10)
residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Mortgaged"])
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Home", "Business", "Education"])

# Manual encodings (must match training)
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
marital_map = {"Single": 2, "Married": 1, "Divorced": 0}
res_map = {"Owned": 2, "Rented": 1, "Mortgaged": 0}
purpose_map = {"Personal": 3, "Car": 1, "Home": 2, "Business": 0, "Education": 4}

# Create input vector
features = np.array([[
    age,
    income,
    loan_amount,
    credit_score,
    employment_years,
    edu_map[education_level],
    marital_map[marital_status],
    num_dependents,
    existing_loans_count,
    res_map[residence_type],
    purpose_map[loan_purpose]
]])

# Scale features
features_scaled = scaler.transform(features)

if st.button("Predict Loan Approval"):
    prediction = model.predict(features_scaled)[0]
    score = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan is likely to be Approved. Score: {score:.2f}")
    else:
        st.error(f"❌ Loan is likely to be Rejected. Score: {score:.2f}")
