from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Global variables for model and data
model = None
x_train = None
y_train = None
x_test = None
y_test = None
training_accuracy = None
testing_accuracy = None

def load_data_and_train_model():
    """Load the sonar data and train the logistic regression model"""
    global model, x_train, y_train, x_test, y_test, training_accuracy, testing_accuracy
    
    # Load the dataset
    df = pd.read_csv('sonar data.csv', header=None)
    
    # Separate features and target
    x = df.drop(columns=60, axis=1)  # Features (60 columns)
    y = df[60]  # Target (Rock/Mine)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, stratify=y, random_state=1
    )
    
    # Train the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # Calculate accuracies
    x_train_prediction = model.predict(x_train)
    training_accuracy = accuracy_score(x_train_prediction, y_train)
    
    x_test_prediction = model.predict(x_test)
    testing_accuracy = accuracy_score(x_test_prediction, y_test)
    
    print(f"Model trained successfully!")
    print(f"Training accuracy: {training_accuracy:.4f}")
    print(f"Testing accuracy: {testing_accuracy:.4f}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get the input data from the request
        data = request.get_json()
        
        # Extract the 60 features
        features = data.get('features', [])
        
        if len(features) != 60:
            return jsonify({
                'error': 'Please provide exactly 60 sonar feature values',
                'success': False
            }), 400
        
        # Convert to numpy array and reshape
        input_data = np.array(features, dtype=float)
        input_data_reshaped = input_data.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_reshaped)[0]
        
        # Get prediction probability
        prediction_proba = model.predict_proba(input_data_reshaped)[0]
        confidence = max(prediction_proba) * 100
        
        # Determine result
        result = "Rock" if prediction == 'R' else "Mine"
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 2),
            'raw_prediction': prediction,
            'probabilities': {
                'Rock': round(prediction_proba[0] * 100, 2),
                'Mine': round(prediction_proba[1] * 100, 2)
            },
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/model_info')
def model_info():
    """Return model information and performance metrics"""
    return jsonify({
        'training_accuracy': round(training_accuracy, 4),
        'testing_accuracy': round(testing_accuracy, 4),
        'training_samples': len(x_train),
        'testing_samples': len(x_test),
        'total_features': 60,
        'model_type': 'Logistic Regression'
    })

@app.route('/sample_data')
def sample_data():
    """Return sample data for testing"""
    # Get a few samples from the test set
    sample_indices = [0, 1, 2]  # First 3 test samples
    samples = []
    
    for idx in sample_indices:
        if idx < len(x_test):
            sample = {
                'features': x_test.iloc[idx].tolist(),
                'actual_label': y_test.iloc[idx],
                'predicted_label': model.predict(x_test.iloc[idx:idx+1])[0]
            }
            samples.append(sample)
    
    return jsonify({'samples': samples})

if __name__ == '__main__':
    # Load data and train model when starting the app
    load_data_and_train_model()
    app.run(debug=True, host='0.0.0.0', port=5000)