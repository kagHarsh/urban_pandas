import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and the scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# CSV file path
csv_file_path = "Urban_Air_Quality_Dataset.csv"

@app.route('/')
def home():
    return render_template('home.html')

# Route to make predictions using API (JSON input)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get JSON data from request
        data = request.json['data']
        
        # Define the order of features as per the dataset
        feature_columns = [
            'tempmax', 'tempmin', 'temp', 
            'feelslikemax', 'feelslikemin', 'dew', 'humidity', 'windspeed', 
            'pressure', 'cloudcover', 'visibility', 'solarradiation'
            , 'uvindex', 'severerisk', 'City', 'Temp_Range', 'Heat_Index', 'Severity_Score'
        ]
        
        # Convert the data into a numpy array
        feature_values = [data[feature] for feature in feature_columns]
        
        # Transform (scale) the input data
        new_data = scalar.transform(np.array(feature_values).reshape(1, -1))
        
        # Predict using the model
        output = regmodel.predict(new_data)
        
        return jsonify(output[0])
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Route for prediction using form input (if applicable)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting data from the form and converting to float
        data = [float(x) for x in request.form.values()]
        
        # Scale and reshape the input data
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        
        # Predict using the model
        output = regmodel.predict(final_input)[0]
        
        return render_template("home.html", prediction_text=f"The health risk score prediction is {output}")
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")



if __name__ == "__main__":
    app.run(debug=True)
