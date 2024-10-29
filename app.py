from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Specify the path to your templates directory
template_dir = os.path.join(os.getcwd(), 'templates')

app = Flask(__name__, template_folder=template_dir)
# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template("web.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the frontend
    input_data = [float(x) for x in request.form.values()]
    features = np.array([input_data])

    # Make prediction
    prediction = model.predict(features)
    
    # Output either 'Heart Disease Present' or 'No Heart Disease'
    result = 'Heart Disease Present' if prediction[0] == 1 else 'No Heart Disease'

    return render_template("web.html", prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)



