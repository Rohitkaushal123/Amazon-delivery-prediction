import streamlit as st
import numpy as np
import pickle
import gdown
import os

# Title
st.title("📦 Amazon Delivery Time Prediction")

# ✅ Download and load the model
model_url = 'https://drive.google.com/uc?id=1RY85dShLmg4QEsIHSUY4yaNtcOV0BtaD'
model_path = 'Random_forest_model.pkl'
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Inputs
st.subheader("Agent & Delivery Details")
age = st.number_input("Agent Age", min_value=18, max_value=60, value=30)
rating = st.slider("Agent Rating", 0.0, 5.0, 4.5)
distance = st.number_input("Distance (km)", min_value=0.0, value=10.0)
order_hour = st.slider("Order Hour (0-23)", 0, 23, 10)
pickup_hour = st.slider("Pickup Hour (0-23)", 0, 23, 12)

st.subheader("Conditions & Metadata")
weather_input = st.selectbox("Weather", ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy'])
traffic_input = st.selectbox("Traffic", ['High', 'Jam', 'Low', 'Medium'])
vehicle_input = st.selectbox("Vehicle", ['bicycle', 'bike', 'car', 'scooter', 'van'])
area_input = st.selectbox("Area", ['Urban', 'Semi-Urban', 'Rural'])
day_input = st.selectbox("Order Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
category_input = st.selectbox("Category", [
    'Apparel', 'Books', 'Clothing', 'Cosmetics', 'Electronics', 'Grocery',
    'Home', 'Jewelry', 'Kitchen', 'Outdoors', 'Pet Supplies', 'Shoes',
    'Skincare', 'Snacks', 'Sports', 'Toys'])

if st.button("Predict Delivery Time"):
    try:
        # One-hot encode categorical inputs
        weather_options = ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy']
        traffic_options = ['High', 'Jam', 'Low', 'Medium']
        vehicle_options = ['bicycle', 'bike', 'car', 'scooter', 'van']
        area_options = ['Urban', 'Semi-Urban', 'Rural']
        day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        category_options = ['Apparel', 'Books', 'Clothing', 'Cosmetics', 'Electronics', 'Grocery',
                            'Home', 'Jewelry', 'Kitchen', 'Outdoors', 'Pet Supplies', 'Shoes',
                            'Skincare', 'Snacks', 'Sports', 'Toys']

        weather_features = [1 if weather_input == w else 0 for w in weather_options]
        traffic_features = [1 if traffic_input == t else 0 for t in traffic_options] + [0]  # traffic_NaN = 0
        vehicle_features = [1 if vehicle_input == v else 0 for v in vehicle_options]
        area_features = [1 if area_input == a else 0 for a in area_options]
        day_features = [1 if day_input == d else 0 for d in day_options]
        category_features = [1 if category_input == c else 0 for c in category_options]

        final_input = [age, rating, distance, order_hour, pickup_hour] + \
                      weather_features + traffic_features + \
                      vehicle_features + area_features + day_features + category_features

        final_input = np.array([final_input])

        prediction = model.predict(final_input)[0]
        st.success(f"🚚 Estimated Delivery Time: {prediction:.2f} minutes")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
