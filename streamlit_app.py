import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('random_forest_model.joblib')  # Assuming model is in the root directory

# Define the app title
st.title('Customer Churn Prediction')

# Create input fields for user data
credit_score = st.number_input('Credit Score', min_value=350, max_value=850, value=650)
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, value=10000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.checkbox('Has Credit Card')
is_active_member = st.checkbox('Is Active Member')
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

# Create a dictionary to store the input data
input_data = {
    'CreditScore': credit_score,
    'Geography_Germany': 1 if geography == 'Germany' else 0,
    'Geography_Spain': 1 if geography == 'Spain' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_cr_card else 0,
    'IsActiveMember': 1 if is_active_member else 0,
    'EstimatedSalary': estimated_salary,
    'CustomerLifespan': age * tenure,
    'Age_IsActiveMember': age * (1 if is_active_member else 0),  # Interaction feature
}

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Make predictions when the user clicks the "Predict" button
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    prediction_probability = model.predict_proba(input_df)[0][1]  # Probability of churn

    if prediction == 1:
        st.write('This customer is likely to churn.')
    else:
        st.write('This customer is likely to stay.')

    st.write(f'Churn Probability: {prediction_probability:.2f}')
