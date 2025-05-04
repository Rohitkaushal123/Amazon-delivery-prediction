import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and columns
model = pickle.load(open('Gradient_Boost_model.pkl', 'rb'))  # If file is at root level



with open(model_path, 'rb') as f:
    model = pickle.load(f)


st.title("📦Amazon Delivery Time Prediction")

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
    # Step 1: Start with 0 for all columns
    input_data = {key: 0 for key in col}

    # Step 2: Fill numeric values
    input_data['Agent_Age'] = age
    input_data['Agent_Rating'] = rating
    input_data['Distance'] = distance
    input_data['Order_Hour'] = order_hour
    input_data['Pickup_Hour'] = pickup_hour

    # Step 3: Fill categorical one-hot encoded values
    traffic_col = f'Traffic_{traffic}'
    weather_col = f'Weather_conditions_{weather}'
    vehicle_col = f'Vehicle_condition_{vehicle}'
    area_col = f'Area_{area}'
    category_col = f'Category_{category}'

    for col_name in [traffic_col, weather_col, vehicle_col, area_col, category_col]:
        if col_name in input_data:
            input_data[col_name] = 1

    # Step 4: Convert to DataFrame
    final_input = pd.DataFrame([input_data])

    # Step 5: Predict
    prediction = model.predict(final_input)[0]
    st.success(f"🚚 Estimated Delivery Time: {round(prediction, 2)} hours")
