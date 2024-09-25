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
    # Load the CSV file
    df = pd.read_csv(csv_file_path, header=0)  # Assuming the first row is the header
    
    # Extract features for prediction from the first row (you can customize the row number)
    feature_columns = df.columns[:-1]  # Adjust to exclude any non-feature columns
    new_data = scalar.transform(df[feature_columns].iloc[0].values.reshape(1, -1))
    
    # Predict using the model
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

# Route for prediction using form input (if applicable)
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data as input
    data = [float(x) for x in request.form.values()]
    
    # Scale and reshape input
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    
    # Predict using the model
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The health risk score prediction is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
