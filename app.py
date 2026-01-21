from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
MODEL_PATH = 'model/house_price_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
ENCODER_PATH = 'model/label_encoder.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    label_encoder = None

# Get list of neighborhoods from label encoder
neighborhoods = list(label_encoder.classes_) if label_encoder else []

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None or label_encoder is None:
            return jsonify({
                'error': 'Model not loaded. Please check model files.'
            }), 500
        
        # Get form data
        overall_qual = int(request.form.get('overall_qual', 5))
        gr_liv_area = float(request.form.get('gr_liv_area', 1500))
        total_bsmt_sf = float(request.form.get('total_bsmt_sf', 1000))
        garage_cars = float(request.form.get('garage_cars', 2))
        year_built = int(request.form.get('year_built', 2000))
        neighborhood = request.form.get('neighborhood', 'NAmes')
        
        # Encode neighborhood
        neighborhood_encoded = label_encoder.transform([neighborhood])[0]
        
        # Prepare features in the same order as training
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': float(prediction),
            'formatted_price': f"${prediction:,.2f}"
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)