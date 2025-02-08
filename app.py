import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Sample data for training the model (replace with your actual data)
data = [
    [40, 25, 20, 30, 5, 10],
    [45, 30, 15, 32, 6, 12],
    [50, 20, 25, 28, 5.5, 11],
    [35, 35, 10, 29, 5.8, 9],
    [42, 28, 20, 31, 5.2, 10.5],
    # Add more samples as needed
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Cellulose (%)', 'Hemicellulose (%)', 'Lignin (%)', 
                                 'Temperature (°C)', 'pH', 'Ethanol Yield (L/ton)'])

# Splitting data into train and test sets
X = df.drop(columns=['Ethanol Yield (L/ton)'])  # Features
y = df['Ethanol Yield (L/ton)']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Machine Learning Model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    cellulose = float(request.form['cellulose'])
    hemicellulose = float(request.form['hemicellulose'])
    lignin = float(request.form['lignin'])
    temperature = float(request.form['temperature'])
    pH = float(request.form['pH'])

    input_data = pd.DataFrame([[cellulose, hemicellulose, lignin, temperature, pH]], 
                              columns=['Cellulose (%)', 'Hemicellulose (%)', 'Lignin (%)', 
                                       'Temperature (°C)', 'pH'])  # Preserve feature names

    predicted_yield = model.predict(input_data)[0]
    return render_template('result.html', predicted_yield=predicted_yield)

if __name__ == '__main__':
    app.run(debug=True)