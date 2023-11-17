import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained Keras model
model = load_model("Churning_Customers.h5")

# Function to preprocess input data
def preprocess_input(input_data, label_encoder_dict=None, scaler=None):
    if label_encoder_dict is None:
        label_encoder_dict = {}
        scaler = StandardScaler()

    columns = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract', 'PaymentMethod']
    input_data = pd.DataFrame(input_data, columns=columns)

    for column in input_data.select_dtypes(include=['object']).columns:
        if column not in label_encoder_dict:
            label_encoder_dict[column] = LabelEncoder()
            input_data[column] = label_encoder_dict[column].fit_transform(input_data[column])

    numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_data[numerical] = scaler.fit_transform(input_data[numerical])
    return input_data

# Function to make predictions
def make_prediction(input_data):
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data)

    # Make predictions using the loaded Keras model
    predictions = model.predict(preprocessed_data)

    return predictions

# Streamlit app
def main():
    st.title("Churn Prediction Web App")

    # Create input widgets (example: numeric input fields)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=200.0)
    tenure = st.number_input("Tenure (months)", min_value=0, value=100)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    # Collect user input into a dictionary
    user_input = {
        "MonthlyCharges": monthly_charges,
        "tenure": tenure,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
    }

    # Make predictions
    if st.button('predict'):
        input_data = np.array([[monthly_charges, tenure, total_charges, contract, payment_method]])

        # Perform prediction
        predictions = make_prediction(input_data)

        # Display the prediction
        churn_prediction = "Yes" if predictions[0] > 0.5 else "No"
        st.write(f"Churn Prediction: {churn_prediction}")

if __name__ == "__main__":
    main()
