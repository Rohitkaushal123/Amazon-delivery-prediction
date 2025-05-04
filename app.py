from flask import Flask, render_template, request
import numpy as np
import pickle
import gdown

app = Flask(__name__)

# ✅ Download model from Google Drive
url = 'https://drive.google.com/uc?id=1RY85dShLmg4QEsIHSUY4yaNtcOV0BtaD'
output = 'Random_forest_model.pkl'
gdown.download(url, output, quiet=False)

# ✅ Load the model
with open('Random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Numeric inputs
        age = float(request.form['Agent_Age'])
        rating = float(request.form['Agent_Rating'])
        distance = float(request.form['Distance'])
        order_hour = int(request.form['Order_hour'])
        pickup_hour = int(request.form['Pickup_hour'])

        # Step 2: One-hot encoded inputs
        weather_input = request.form['Weather']
        weather_options = ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy']
        weather_features = [1 if weather_input == w else 0 for w in weather_options]

        traffic_input = request.form['Traffic']
        traffic_options = ['High', 'Jam', 'Low', 'Medium']
        traffic_features = [1 if traffic_input == t else 0 for t in traffic_options]
        traffic_nan = [0]  # extra column added during encoding

        vehicle_input = request.form['Vehicle']
        vehicle_options = ['bicycle', 'bike', 'car', 'scooter', 'van']
        vehicle_features = [1 if vehicle_input == v else 0 for v in vehicle_options]

        area_input = request.form['Area']
        area_options = ['Urban', 'Semi-Urban', 'Rural']
        area_features = [1 if area_input == a else 0 for a in area_options]

        day_input = request.form['Order_day']
        day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_features = [1 if day_input == d else 0 for d in day_options]

        category_input = request.form['Category']
        category_options = ['Apparel', 'Books', 'Clothing', 'Cosmetics', 'Electronics', 'Grocery',
                            'Home', 'Jewelry', 'Kitchen', 'Outdoors', 'Pet Supplies', 'Shoes',
                            'Skincare', 'Snacks', 'Sports', 'Toys']
        category_features = [1 if category_input == c else 0 for c in category_options]

        # Final input
        final_input = [age, rating, distance, order_hour, pickup_hour] + \
                      weather_features + traffic_features + traffic_nan + \
                      vehicle_features + area_features + day_features + category_features

        final_input = np.array([final_input])

        # Predict
        prediction = model.predict(final_input)[0]
        return render_template('form.html', prediction_text=f'Predicted Delivery Time: {prediction:.2f} minutes')
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
