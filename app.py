from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained linear regression model
with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Assuming the original feature names
feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_values = [float(request.form.get(name, 0)) for name in feature_names]

    # Make a prediction using the trained model
    prediction = model.predict([input_values])[0]

    # Display the prediction on a result page
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
