from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

model_path = os.path.join('delivery_app', 'Random_forest_model.pkl')
model = pickle.load(open(model_path, 'rb'))


@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Get numerical inputs
        age = float(request.form['Agent_Age'])
        rating = float(request.form['Agent_Rating'])
        distance = float(request.form['Distance'])
        order_hour = int(request.form['Order_hour'])
        pickup_hour = int(request.form['Pickup_hour'])

        # Step 2: Weather one-hot encoding
        weather_input = request.form['Weather']
        weather_options = ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy']
        weather_features = [1 if weather_input == w else 0 for w in weather_options]

        # Step 3: Traffic one-hot encoding
        traffic_input = request.form['Traffic']
        traffic_options = ['High', 'Jam', 'Low', 'Medium']
        traffic_features = [1 if traffic_input == t else 0 for t in traffic_options]
        # ✅ Step 3.5: Add Traffic_NaN manually
        traffic_nan = [0]

        # Step 4: Vehicle one-hot encoding
        vehicle_input = request.form['Vehicle']
        vehicle_options = ['bicycle', 'bike', 'car', 'scooter', 'van']
        vehicle_features = [1 if vehicle_input == v else 0 for v in vehicle_options]

        # Step 5: Area one-hot encoding
        area_input = request.form['Area']
        area_options = ['Urban', 'Semi-Urban', 'Rural']
        area_features = [1 if area_input == a else 0 for a in area_options]

        # Step 6: Day one-hot encoding
        day_input = request.form['Order_day']
        day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_features = [1 if day_input == d else 0 for d in day_options]

        # Step 7: Category one-hot encoding
        category_input = request.form['Category']
        category_options = ['Apparel', 'Books', 'Clothing', 'Cosmetics', 'Electronics', 'Grocery',
                            'Home', 'Jewelry', 'Kitchen', 'Outdoors', 'Pet Supplies', 'Shoes',
                            'Skincare', 'Snacks', 'Sports', 'Toys']
        category_features = [1 if category_input == c else 0 for c in category_options]


        final_input = [age, rating, distance, order_hour, pickup_hour] + \
              weather_features + traffic_features + traffic_nan + \
              vehicle_features + area_features + day_features + category_features

        final_input = np.array([final_input])

        # Step 5: Predict
        prediction = model.predict(final_input)[0]
        return render_template('form.html', prediction_text=f'Predicted Delivery Time: {prediction:.2f} minutes')
    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
