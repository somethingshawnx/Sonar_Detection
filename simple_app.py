#!/usr/bin/env python3
"""
Simplified Sonar Rock vs Mine Detection Web Application
This version works with basic Python installation
"""

import json
import math
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Simple logistic regression implementation
class SimpleLogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.trained = False
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 0.0 if z < 0 else 1.0
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        """Simple gradient descent training"""
        n_samples, n_features = len(X), len(X[0])
        
        # Initialize weights and bias
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = random.uniform(-0.1, 0.1)
        
        # Convert labels to binary (R=0, M=1)
        y_binary = [1 if label == 'M' else 0 for label in y]
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            predictions = []
            for i in range(n_samples):
                z = sum(self.weights[j] * X[i][j] for j in range(n_features)) + self.bias
                pred = self.sigmoid(z)
                predictions.append(pred)
            
            # Compute gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = predictions[i] - y_binary[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= learning_rate * dw[j] / n_samples
            self.bias -= learning_rate * db / n_samples
        
        self.trained = True
    
    def predict(self, X):
        """Make predictions"""
        if not self.trained:
            return ['R'] * len(X)
        
        predictions = []
        for sample in X:
            z = sum(self.weights[j] * sample[j] for j in range(len(sample))) + self.bias
            pred = self.sigmoid(z)
            predictions.append('M' if pred > 0.5 else 'R')
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.trained:
            return [[0.5, 0.5]] * len(X)
        
        probabilities = []
        for sample in X:
            z = sum(self.weights[j] * sample[j] for j in range(len(sample))) + self.bias
            prob_mine = self.sigmoid(z)
            prob_rock = 1 - prob_mine
            probabilities.append([prob_rock, prob_mine])
        return probabilities

# Load and prepare data
def load_data():
    """Load sonar data from CSV"""
    data = []
    labels = []
    
    try:
        with open('sonar data.csv', 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) == 61:  # 60 features + 1 label
                        features = [float(x) for x in parts[:60]]
                        label = parts[60]
                        data.append(features)
                        labels.append(label)
    except FileNotFoundError:
        print("Warning: sonar data.csv not found. Using dummy data.")
        # Generate dummy data for demonstration
        for _ in range(100):
            features = [random.uniform(0, 1) for _ in range(60)]
            label = 'M' if random.random() > 0.5 else 'R'
            data.append(features)
            labels.append(label)
    
    return data, labels

# Global model
model = SimpleLogisticRegression()
training_data = None
training_labels = None
test_data = None
test_labels = None
training_accuracy = 0.0
testing_accuracy = 0.0

def train_model():
    """Train the model"""
    global model, training_data, training_labels, test_data, test_labels
    global training_accuracy, testing_accuracy
    
    # Load data
    data, labels = load_data()
    
    # Simple train-test split (80-20)
    split_idx = int(0.8 * len(data))
    training_data = data[:split_idx]
    training_labels = labels[:split_idx]
    test_data = data[split_idx:]
    test_labels = labels[split_idx:]
    
    # Train model
    model.fit(training_data, training_labels)
    
    # Calculate accuracies
    train_pred = model.predict(training_data)
    training_accuracy = sum(1 for i in range(len(training_labels)) 
                           if train_pred[i] == training_labels[i]) / len(training_labels)
    
    test_pred = model.predict(test_data)
    testing_accuracy = sum(1 for i in range(len(test_labels)) 
                          if test_pred[i] == test_labels[i]) / len(test_labels)
    
    print(f"Model trained! Training accuracy: {training_accuracy:.3f}, Testing accuracy: {testing_accuracy:.3f}")

class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_html()
        elif self.path == '/model_info':
            self.serve_model_info()
        elif self.path == '/sample_data':
            self.serve_sample_data()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/predict':
            self.handle_predict()
        else:
            self.send_error(404)
    
    def serve_html(self):
        """Serve the main HTML page"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sonar Rock vs Mine Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; 
            color: #333; 
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; color: white; }
        .header h1 { font-size: 3rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }
        .card { 
            background: white; border-radius: 20px; padding: 30px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
        }
        .feature-inputs { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); 
            gap: 10px; margin-bottom: 20px; max-height: 400px; overflow-y: auto; 
            padding: 10px; border: 2px dashed #e2e8f0; border-radius: 10px; 
        }
        .feature-input { display: flex; flex-direction: column; align-items: center; }
        .feature-input label { font-size: 0.8rem; color: #718096; margin-bottom: 5px; }
        .feature-input input { 
            width: 100%; padding: 8px; border: 2px solid #e2e8f0; border-radius: 8px; 
            text-align: center; 
        }
        .feature-input input:focus { outline: none; border-color: #667eea; }
        .buttons { display: flex; gap: 15px; margin-top: 20px; }
        .btn { 
            padding: 12px 24px; border: none; border-radius: 10px; 
            font-size: 1rem; font-weight: 600; cursor: pointer; 
        }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; flex: 1; }
        .btn-secondary { background: #f7fafc; color: #4a5568; border: 2px solid #e2e8f0; }
        .result-card { 
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
            color: white; text-align: center; padding: 30px; border-radius: 15px; 
            margin-bottom: 20px; display: none; 
        }
        .result-card.rock { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .result-card.mine { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        .result-icon { font-size: 3rem; margin-bottom: 15px; }
        .result-text { font-size: 1.5rem; font-weight: 700; margin-bottom: 10px; }
        .confidence { font-size: 1rem; opacity: 0.9; }
        .probabilities { display: flex; justify-content: space-around; margin-top: 15px; }
        .prob-item { text-align: center; }
        .prob-value { font-size: 1.2rem; font-weight: 600; }
        .prob-label { font-size: 0.9rem; opacity: 0.8; }
        .model-info { background: #f8fafc; border-radius: 10px; padding: 20px; margin-top: 20px; }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .info-item { text-align: center; }
        .info-value { font-size: 1.5rem; font-weight: 700; color: #667eea; }
        .info-label { font-size: 0.9rem; color: #718096; margin-top: 5px; }
        .error { background: #fed7d7; color: #c53030; padding: 15px; border-radius: 10px; margin-top: 15px; display: none; }
        .sample-data { background: #e6fffa; border: 2px solid #81e6d9; border-radius: 10px; padding: 15px; margin-top: 15px; }
        .sample-btn { 
            background: #38b2ac; color: white; border: none; padding: 8px 16px; 
            border-radius: 6px; cursor: pointer; margin-right: 10px; margin-bottom: 10px; 
        }
        .sample-btn:hover { background: #319795; }
        @media (max-width: 768px) { .main-content { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Sonar Detection System</h1>
            <p>AI-Powered Rock vs Mine Classification using Logistic Regression</p>
        </div>

        <div class="main-content">
            <div class="card">
                <h2>📊 Input Sonar Data</h2>
                <p style="color: #718096; margin-bottom: 20px;">
                    Enter 60 sonar feature values (0.0 to 1.0) to classify the object as Rock or Mine
                </p>
                
                <div class="feature-inputs" id="featureInputs">
                    <!-- Feature inputs will be generated by JavaScript -->
                </div>

                <div class="buttons">
                    <button class="btn btn-primary" onclick="predict()">🔍 Classify Object</button>
                    <button class="btn btn-secondary" onclick="clearInputs()">🧹 Clear</button>
                </div>

                <div class="error" id="error"></div>

                <div class="sample-data">
                    <h4>🧪 Try Sample Data</h4>
                    <p style="color: #234e52; margin-bottom: 10px; font-size: 0.9rem;">
                        Click a button below to load sample data for testing
                    </p>
                    <button class="sample-btn" onclick="loadSampleData(0)">Sample 1 (Rock)</button>
                    <button class="sample-btn" onclick="loadSampleData(1)">Sample 2 (Mine)</button>
                    <button class="sample-btn" onclick="loadSampleData(2)">Sample 3 (Mixed)</button>
                </div>
            </div>

            <div class="card">
                <h2>📈 Classification Result</h2>
                
                <div class="result-card" id="resultCard">
                    <div class="result-icon" id="resultIcon"></div>
                    <div class="result-text" id="resultText"></div>
                    <div class="confidence" id="confidence"></div>
                    <div class="probabilities" id="probabilities"></div>
                </div>

                <div class="model-info">
                    <h3>📊 Model Performance</h3>
                    <div class="info-grid" id="modelInfo">
                        <!-- Model info will be loaded by JavaScript -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize the feature inputs
        function initializeInputs() {
            const container = document.getElementById('featureInputs');
            for (let i = 0; i < 60; i++) {
                const inputDiv = document.createElement('div');
                inputDiv.className = 'feature-input';
                inputDiv.innerHTML = `
                    <label>F${i + 1}</label>
                    <input type="number" id="feature${i}" min="0" max="1" step="0.0001" placeholder="0.0000">
                `;
                container.appendChild(inputDiv);
            }
        }

        // Load sample data
        async function loadSampleData(sampleIndex) {
            try {
                const response = await fetch('/sample_data');
                const data = await response.json();
                
                if (data.samples && data.samples[sampleIndex]) {
                    const sample = data.samples[sampleIndex];
                    for (let i = 0; i < 60; i++) {
                        document.getElementById(`feature${i}`).value = sample.features[i].toFixed(4);
                    }
                    showError('');
                }
            } catch (error) {
                showError('Failed to load sample data: ' + error.message);
            }
        }

        // Clear all inputs
        function clearInputs() {
            for (let i = 0; i < 60; i++) {
                document.getElementById(`feature${i}`).value = '';
            }
            hideResult();
            showError('');
        }

        // Show error message
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = message ? 'block' : 'none';
        }

        // Hide result card
        function hideResult() {
            document.getElementById('resultCard').style.display = 'none';
        }

        // Show result card
        function showResult(prediction, confidence, probabilities) {
            const resultCard = document.getElementById('resultCard');
            const resultIcon = document.getElementById('resultIcon');
            const resultText = document.getElementById('resultText');
            const confidenceDiv = document.getElementById('confidence');
            const probabilitiesDiv = document.getElementById('probabilities');

            if (prediction === 'Rock') {
                resultCard.className = 'result-card rock';
                resultIcon.textContent = '🏔️';
                resultText.textContent = 'ROCK DETECTED';
            } else {
                resultCard.className = 'result-card mine';
                resultIcon.textContent = '💣';
                resultText.textContent = 'MINE DETECTED';
            }

            confidenceDiv.textContent = `Confidence: ${confidence}%`;
            
            probabilitiesDiv.innerHTML = `
                <div class="prob-item">
                    <div class="prob-value">${probabilities.Rock}%</div>
                    <div class="prob-label">Rock</div>
                </div>
                <div class="prob-item">
                    <div class="prob-value">${probabilities.Mine}%</div>
                    <div class="prob-label">Mine</div>
                </div>
            `;

            resultCard.style.display = 'block';
        }

        // Make prediction
        async function predict() {
            const features = [];
            let isValid = true;
            
            for (let i = 0; i < 60; i++) {
                const value = document.getElementById(`feature${i}`).value;
                if (value === '' || isNaN(value) || value < 0 || value > 1) {
                    isValid = false;
                    break;
                }
                features.push(parseFloat(value));
            }

            if (!isValid) {
                showError('Please enter valid values (0.0 to 1.0) for all 60 features');
                return;
            }

            hideResult();
            showError('');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: features })
                });

                const data = await response.json();

                if (data.success) {
                    showResult(data.prediction, data.confidence, data.probabilities);
                } else {
                    showError(data.error || 'Prediction failed');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            }
        }

        // Load model information
        async function loadModelInfo() {
            try {
                const response = await fetch('/model_info');
                const data = await response.json();
                
                const modelInfoDiv = document.getElementById('modelInfo');
                modelInfoDiv.innerHTML = `
                    <div class="info-item">
                        <div class="info-value">${(data.training_accuracy * 100).toFixed(1)}%</div>
                        <div class="info-label">Training Accuracy</div>
                    </div>
                    <div class="info-item">
                        <div class="info-value">${(data.testing_accuracy * 100).toFixed(1)}%</div>
                        <div class="info-label">Testing Accuracy</div>
                    </div>
                    <div class="info-item">
                        <div class="info-value">${data.training_samples}</div>
                        <div class="info-label">Training Samples</div>
                    </div>
                    <div class="info-item">
                        <div class="info-value">${data.testing_samples}</div>
                        <div class="info-label">Testing Samples</div>
                    </div>
                    <div class="info-item">
                        <div class="info-value">60</div>
                        <div class="info-label">Features</div>
                    </div>
                    <div class="info-item">
                        <div class="info-value">Logistic Regression</div>
                        <div class="info-label">Algorithm</div>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load model info:', error);
            }
        }

        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            initializeInputs();
            loadModelInfo();
        });
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_model_info(self):
        """Serve model information"""
        global training_accuracy, testing_accuracy, training_data, test_data
        
        info = {
            'training_accuracy': training_accuracy,
            'testing_accuracy': testing_accuracy,
            'training_samples': len(training_data) if training_data else 0,
            'testing_samples': len(test_data) if test_data else 0,
            'total_features': 60,
            'model_type': 'Logistic Regression'
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(info).encode())
    
    def serve_sample_data(self):
        """Serve sample data for testing"""
        global test_data, test_labels
        
        samples = []
        if test_data and test_labels:
            for i in range(min(3, len(test_data))):
                sample = {
                    'features': test_data[i],
                    'actual_label': test_labels[i],
                    'predicted_label': model.predict([test_data[i]])[0]
                }
                samples.append(sample)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'samples': samples}).encode())
    
    def handle_predict(self):
        """Handle prediction requests"""
        global model
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            features = data.get('features', [])
            
            if len(features) != 60:
                response = {
                    'error': 'Please provide exactly 60 sonar feature values',
                    'success': False
                }
            else:
                # Make prediction
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
                confidence = max(probabilities) * 100
                
                result = "Rock" if prediction == 'R' else "Mine"
                
                response = {
                    'prediction': result,
                    'confidence': round(confidence, 2),
                    'raw_prediction': prediction,
                    'probabilities': {
                        'Rock': round(probabilities[0] * 100, 2),
                        'Mine': round(probabilities[1] * 100, 2)
                    },
                    'success': True
                }
            
        except Exception as e:
            response = {
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

def main():
    """Main function to start the server"""
    print("🚀 Starting Sonar Rock vs Mine Detection Web App")
    print("=" * 50)
    
    # Train the model
    print("📊 Training model...")
    train_model()
    
    # Start the server
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, WebHandler)
    
    print("🌐 Web interface available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
        httpd.server_close()

if __name__ == '__main__':
    main()