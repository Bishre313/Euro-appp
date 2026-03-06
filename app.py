from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline
model = joblib.load("eurovision_model.pkl")

# Load your CSV file (matches your folder)
voting_data = pd.read_csv("Voting Final.csv")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = [
            data['danceability'],
            data['acousticness'],
            data['speechiness'],
            data['tempo'],
            data['valence']
        ]
        prediction = model.predict(np.array(features).reshape(1, -1))
        return jsonify({'predicted_points': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/voting/<country>', methods=['GET'])
def voting_analysis(country):
    try:
        # Filter rows for the selected country
        filtered = voting_data[voting_data['Country'].str.lower() == country.lower()]
        if filtered.empty:
            return f"No voting data found for {country}."

        # Sum points across all years
        total_points = filtered['Points'].sum()

        return f"{country} has historically received a total of {total_points} points."
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
