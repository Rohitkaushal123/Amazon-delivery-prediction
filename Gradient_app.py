import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
model_path = 'Gradient_Boost_model.pkl'  # This is the actual file path
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define all columns expected by the model
col = [
    'Agent_Age', 'Agent_Rating', 'Distance', 'Order_Hour', 'Pickup_Hour',
    'Traffic_Low', 'Traffic_Medium', 'Traffic_High', 'Traffic_Jam',
    'Weather_conditions_Sunny', 'Weather_conditions_Stormy', 'Weather_conditions_Sandstorms',
    'Weather_conditions_Cloudy', 'Weather_conditions_Windy', 'Weather_conditions_Fog',
    'Vehicle_condition_Scooter', 'Vehicle_condition_Bike', 'Vehicle_condition_Car',
    'Area_Urban', 'Area_Metropolitan',
    'Category_Laptop', 'Category_Mobile', 'Category_Fashion', 'Category_Grocery', 'Category_Others'
]

st.title("📦 Amazon Delivery Time Prediction")

# User Inputs
age = st.number_input("Agent Age", min_value=18, max_value=60, value=30)
rating = st.slider("Agent Rating", 0.0, 5.0, 4.5)
distance = st.number_input("Distance (km)", min_value=0.0, value=10.0)
order_hour = st.slider("Order Hour (0–23)", 0, 23, 10)
pickup_hour = st.slider("Pickup Hour (0–23)", 0, 23, 12)

traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
weather = st.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Windy", "Fog"])
vehicle = st.selectbox("Vehicle", ["Scooter", "Bike", "Car"])
area = st.selectbox("Area", ["Urban", "Metropolitan"])
category = st.selectbox("Category", ["Laptop", "Mobile", "Fashion", "Grocery", "Others"])

# Prediction button
if st.button("Predict Delivery Time"):
    # Step 1: Start with all zeros
    input_data = {key: 0 for key in col}

    # Step 2: Assign numerical values
    input_data['Agent_Age'] = age
    input_data['Agent_Rating'] = rating
    input_data['Distance'] = distance
    input_data['Order_Hour'] = order_hour
    input_data['Pickup_Hour'] = pickup_hour

    # Step 3: Set one-hot values
    feature_map = {
        f'Traffic_{traffic}': 'traffic',
        f'Weather_conditions_{weather}': 'weather',
        f'Vehicle_condition_{vehicle}': 'vehicle',
        f'Area_{area}': 'area',
        f'Category_{category}': 'category'
    }

    for feature in feature_map:
        if feature in input_data:
            input_data[feature] = 1

    # Step 4: Convert to DataFrame
    final_input = pd.DataFrame([input_data])

    # Step 5: Predict
    prediction = model.predict(final_input)[0]
    st.success(f"🚚 Estimated Delivery Time: {round(prediction, 2)} hours")
