import streamlit as st

from src.predict import predict_churn


st.set_page_config(page_title="Customer Churn Prediction System", layout="centered")


def main():
    st.title("Customer Churn Prediction System")

    st.markdown(
        "Use the form below to input customer details and predict the probability of churn."
    )

    with st.form("churn_form"):
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, max_value=10000.0, value=840.0, step=10.0
        )

        contract = st.selectbox(
            "Contract",
            options=["Month-to-month", "One year", "Two year"],
        )

        internet_service = st.selectbox(
            "Internet Service",
            options=["DSL", "Fiber optic", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            prediction, probability = predict_churn(
                tenure=tenure,
                monthly_charges=monthly_charges,
                total_charges=total_charges,
                contract=contract,
                internet_service=internet_service,
                payment_method=payment_method,
            )

            if prediction == 1:
                st.error(
                    f"The model predicts that the customer is **likely to churn**.\n\n"
                    f"Churn probability: **{probability:.2%}**"
                )
            else:
                st.success(
                    f"The model predicts that the customer is **not likely to churn**.\n\n"
                    f"Churn probability: **{probability:.2%}**"
                )
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred while making prediction: {e}")


if __name__ == "__main__":
    main()

