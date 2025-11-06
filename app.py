# -----------------------------------------------------------------
# app.py (with Confidence Scores)
# -----------------------------------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np 
import warnings
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  

# --- 1. Load All Models and Scalers ---
try:
    # --- Parkinson's Model ---
    pk_model = joblib.load('parkinsons_model.joblib')
    pk_scaler = joblib.load('parkinsons_scaler.joblib')
    print("[INFO] Parkinson's model and scaler loaded.")
    
    # --- Brain Tumor Model ---
    bt_model = joblib.load('brain_tumor_model.joblib')
    bt_scaler = joblib.load('brain_tumor_scaler.joblib')
    print("[INFO] Brain Tumor model and scaler loaded.")

    
    # --- Alzheimer's Model ---
    az_model = joblib.load('alzheimers_model.joblib')
    az_scaler = joblib.load('alzheimers_scaler.joblib')
    ses_imputer = joblib.load('ses_imputer.joblib')
    mmse_imputer = joblib.load('mmse_imputer.joblib')
    print("[INFO] Alzheimer's model, scaler, and imputers loaded.")

except FileNotFoundError as e:
    print(f"[ERROR] A required model or scaler file was not found: {e}")
    pk_model, pk_scaler, bt_model, bt_scaler = None, None, None, None
    az_model, az_scaler, ses_imputer, mmse_imputer = None, None, None, None

# --- 2. Define Feature Lists for Each Model ---

# (Features lists are unchanged... )
PARKINSONS_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

BRAIN_TUMOR_FEATURES = [
    'Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness',
    'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity',
    'Dissimilarity', 'Correlation', 'Coarseness'
]

ALZHEIMERS_FEATURES = [
    'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'
]

# --- 3. Define API Endpoints ---

@app.route('/', methods=['GET'])
def home():
    # (Home endpoint is unchanged...)
    return jsonify({
        'status': 'Online',
        'message': 'ML Model Server is running.',
        'endpoints': {
            'parkinsons': 'POST /predict_parkinsons',
            'brain_tumor': 'POST /predict_braintumor',
            'alzheimers': 'POST /predict_alzheimers'
        }
    })


@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    """Endpoint for Parkinson's prediction."""
    
    if not pk_model or not pk_scaler:
        return jsonify({'error': 'Parkinson\'s model is not loaded.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided.'}), 400

    try:
        features = [data[key] for key in PARKINSONS_FEATURES]
        features_array = np.array([features])
        features_scaled = pk_scaler.transform(features_array)
        
        # Get probabilities: e.g., [[0.1, 0.9]] (10% Class 0, 90% Class 1)
        probabilities = pk_model.predict_proba(features_scaled)
        
        confidence = float(np.max(probabilities[0])) # confidence score
        
        # Get the index of the highest prob as the prediction
        prediction_result = int(np.argmax(probabilities[0])) 
        
        label = "Parkinson's Positive" if prediction_result == 1 else "Parkinson's Negative"

        return jsonify({
            'model': 'parkinsons_detector',
            'prediction': prediction_result,
            'label': label,
            'confidence': confidence 
        })

    except KeyError as e:
        return jsonify({'error': f'Missing feature in JSON: {e}'}), 400 #400 error number
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'}), 500


@app.route('/predict_braintumor', methods=['POST'])
def predict_braintumor():
    """Endpoint for Brain Tumor prediction."""
    
    if not bt_model or not bt_scaler:
        return jsonify({'error': 'Brain Tumor model is not loaded.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided.'}), 400

    try:
        features = [data[key] for key in BRAIN_TUMOR_FEATURES]
        features_array = np.array([features])
        features_scaled = bt_scaler.transform(features_array)

        # Get probabilities: e.g., [[0.95, 0.05]]
        probabilities = bt_model.predict_proba(features_scaled)
        
        # Get the highest probability as the confidence score
        confidence = float(np.max(probabilities[0]))
        
        # Get the index of the highest prob as the prediction
        prediction_result = int(np.argmax(probabilities[0]))
        
        label = "Tumor" if prediction_result == 1 else "No Tumor"
        
        return jsonify({
            'model': 'brain_tumor_detector',
            'prediction': prediction_result,
            'label': label,
            'confidence': confidence 
        })

    except KeyError as e:
        return jsonify({'error': f'Missing feature in JSON: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'}), 500
    
@app.route('/predict_alzheimers', methods=['POST'])
def predict_alzheimers():
    """Endpoint for Alzheimer's prediction."""
    
    # Check if model is loaded
    if not all([az_model, az_scaler, ses_imputer, mmse_imputer]):
        return jsonify({'error': 'Alzheimer\'s model is not loaded.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided.'}), 400

    try:
        # --- 1. Get data in correct order ---
        features = [data[key] for key in ALZHEIMERS_FEATURES]
        features_array = np.array([features])
        
        # --- 2. Create DataFrame for processing ---
        X_new = pd.DataFrame(features_array, columns=ALZHEIMERS_FEATURES)

        # --- 3. Handle Missing Data (Imputation) ---
        # Use the *saved imputers* from training
        X_new['SES'] = ses_imputer.transform(X_new[['SES']])
        X_new['MMSE'] = mmse_imputer.transform(X_new[['MMSE']])
        
        # --- 4. Handle Categorical Data ---
        # Map 'M/F' (which is 0 or 1 from the form)
        # Ensure it's a number, not string
        X_new['M/F'] = X_new['M/F'].astype(float)
        
        # --- 5. Scale and Predict ---
        features_scaled = az_scaler.transform(X_new)
        
        probabilities = az_model.predict_proba(features_scaled)
        confidence = float(np.max(probabilities[0]))
        prediction_result = int(np.argmax(probabilities[0]))
        
        label = "Demented" if prediction_result == 1 else "Nondemented"

        return jsonify({
            'model': 'alzheimers_detector',
            'prediction': prediction_result,
            'label': label,
            'confidence': confidence
        })

    except KeyError as e:
        return jsonify({'error': f'Missing feature in JSON: {e}'}), 400
    except Exception as e:
        # Log the error for debugging
        print(f"[ERROR] in /predict_alzheimers: {e}")
        return jsonify({'error': f'Error during prediction: {e}'}), 500
    

# --- 4. Run the Application ---
if __name__ == '__main__':
    print("[INFO] Starting Flask server with confidence scores enabled...")
    app.run(host='0.0.0.0', port=5000, debug=True)