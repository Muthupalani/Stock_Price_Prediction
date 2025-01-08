from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("stock_price_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process input and predict
    prediction = model.predict(np.array([data['input']]))
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
