import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

model = joblib.load("loan_approval_model.pkl")
threshold = joblib.load("best_threshold.pkl")

st.title("🏦 Loan Approval Prediction")
st.markdown(
    "Predict whether a loan application is likely to be **Approved** or **Rejected** using a Machine Learning model."
)

st.sidebar.header("ℹ️ About")

st.sidebar.info(
    """
    **Model:** Logistic Regression
    
    **Dataset:** Loan Approval Dataset
    
    **Pipeline:**
    - StandardScaler
    - SMOTE
    - Logistic Regression
    
    **Threshold:** {:.2f}
    """.format(threshold)
)

st.header("📋 Applicant Information")

gender = st.selectbox("Gender", ["Male", "Female"])

married = st.selectbox("Married", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

loan_amount = st.number_input(
    "Loan Amount (in thousands)",
    min_value=0,
    value=120
)

loan_term = st.selectbox(
    "Loan Amount Term",
    [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
)

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

input_data = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history
}

input_data["Dependents_1"] = 1 if dependents == "1" else 0
input_data["Dependents_2"] = 1 if dependents == "2" else 0
input_data["Dependents_3+"] = 1 if dependents == "3+" else 0

input_data["Property_Area_Semiurban"] = 1 if property_area == "Semiurban" else 0
input_data["Property_Area_Urban"] = 1 if property_area == "Urban" else 0

input_df = pd.DataFrame([input_data])

if st.button("🔍 Predict Loan Status",
             use_container_width=True):
        probability = model.predict_proba(input_df)[0][1]

        prediction = 1 if probability >= threshold else 0

        st.subheader("Prediction Result")       

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")


        probabilities = model.predict_proba(input_df)[0]

        approval_prob = probabilities[1]
        rejection_prob = probabilities[0]

        st.metric(
            "Approval Probability",
            f"{approval_prob*100:.2f}%"
        )

        st.metric(
            "Rejection Probability",
            f"{rejection_prob*100:.2f}%"
        )

        # ================= Feature Importance ================= #

        st.subheader("📊 Feature contribution for this prediction   ")

        # Extract the trained Logistic Regression model
        lr_model = model.named_steps["model"]

        # Get model coefficients
        coefficients = lr_model.coef_[0]

        # Calculate contribution of each feature
        contributions = input_df.iloc[0] * coefficients

        # Create DataFrame
        importance_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Contribution": contributions
        })

        # Sort by absolute contribution
        importance_df["Absolute"] = importance_df["Contribution"].abs()
        importance_df = importance_df.sort_values(
            by="Absolute",
            ascending=False
        )

        # Assign colors
        colors = [
            "green" if value > 0 else "red"
            for value in importance_df["Contribution"]
        ]

        # Dynamic title
        title = (
            "Prediction Explaination"
            if prediction == 1
            else "Top Factors Leading to Loan Rejection"
        )

        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.barh(
            importance_df["Feature"],
            importance_df["Contribution"],
            color=colors
        )

        # Vertical line separating positive and negative contributions
        ax.axvline(0, color="black", linewidth=1)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Contribution to Prediction")
        ax.set_ylabel("Features")

        ax.invert_yaxis()

        plt.tight_layout()

        st.pyplot(fig)

        # Small explanation
        st.caption(
            "🟢 Positive values increase the likelihood of loan approval, "
            "while 🔴 negative values decrease it. "
            "The magnitude of each bar indicates how strongly that feature "
            "influenced the model's prediction for this applicant."
        )