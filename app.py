from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model
model = load_model('model/stock_price_model.h5')

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol']
    date_range = request.form['date_range']
    features = request.form['features']

    # Dummy data processing (replace with actual implementation)
    # For simplicity, this generates random predictions for now
    test_data = np.random.rand(60, 1)  # Simulate 60 days of stock prices
    test_data = test_data.reshape(1, 60, 1)  # Reshape to match LSTM input

    # Make prediction
    prediction = model.predict(test_data)
    predicted_price = round(float(prediction[0, 0]), 2)

    # Render the result back to the user
    return render_template('index.html', prediction=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
