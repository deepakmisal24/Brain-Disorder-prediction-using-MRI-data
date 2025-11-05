import numpy as np
import joblib  # Used to load your pre-trained model and scaler
from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows your frontend to talk to this backend

# --- 1. INITIALIZE THE FLASK APP ---
app = Flask(__name__)
# Apply CORS: This is crucial for allowing your HTML file (on a different "server")
# to make requests to this Python server.
CORS(app) 

# --- 2. LOAD THE PRE-TRAINED "BRAINS" ---
# We load the model, scaler, and column list *once* when the server starts.
# This is efficient. We don't re-load them for every prediction.

model_filename = 'parkinsons_model.joblib'
scaler_filename = 'data_scaler.joblib'
columns_filename = 'feature_columns.txt'

try:
    # Load the trained model
    model = joblib.load(model_filename)
    print(f"Successfully loaded model from {model_filename}")
    
    # Load the fitted scaler
    scaler = joblib.load(scaler_filename)
    print(f"Successfully loaded scaler from {scaler_filename}")
    
    # Load the list of feature names (in the correct order)
    with open(columns_filename, 'r') as f:
        model_columns = [line.strip() for line in f.readlines()]
    print(f"Successfully loaded {len(model_columns)} feature columns from {columns_filename}")

except FileNotFoundError as e:
    print(f"Error: Could not find a required file. {e}")
    print("Please make sure 'parkinsons_model.joblib', 'data_scaler.joblib', and 'feature_columns.txt' are in the same directory as app.py")
    model = None  # Set to None to prevent the app from running broken
    scaler = None
    model_columns = None
except Exception as e:
    print(f"An error occurred during loading: {e}")
    model = None
    scaler = None
    model_columns = None


# --- 3. DEFINE THE PREDICTION ENDPOINT ---
# This function will run every time the frontend sends data to the '/predict' URL.
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the models loaded correctly at startup
    if not model or not scaler or not model_columns:
        # 500 = Internal Server Error
        return jsonify({'error': 'Model or scaler is not loaded. Check server logs.'}), 500

    try:
        # 1. Get the new data from the user
        # 'request.get_json()' automatically parses the JSON data sent by the frontend
        json_data = request.get_json()

        # 2. Prepare the data for the model
        # We must ensure the 22 features are in the *exact same order*
        # as when the model was trained. Our 'model_columns' list handles this.
        
        ordered_input_data = []
        for col_name in model_columns:
            # Get the value from the JSON data using the column name
            ordered_input_data.append(json_data[col_name])

        # 3. Convert data to a 2D numpy array
        # scikit-learn models always expect a 2D array, even for a single prediction.
        # This creates an array like: [[val1, val2, val3, ...]]
        input_array = np.array([ordered_input_data])

        # 4. Use the loaded scaler and model
        # IMPORTANT: We use .transform() here, NOT .fit() or .fit_transform()
        # We want to apply the *same* scaling rules we learned from the training data.
        input_scaled = scaler.transform(input_array)
        
        # Make the prediction (will be 0 or 1)
        prediction_value = model.predict(input_scaled)
        
        # Get the prediction confidence (optional, but very cool)
        # This gives a [prob_of_0, prob_of_1]
        probabilities = model.predict_proba(input_scaled)
        
        # Get the confidence score for the *predicted* class
        predicted_class_index = int(prediction_value[0])
        confidence_score = float(probabilities[0][predicted_class_index])

        # 5. Send the result back to the frontend
        # We must convert numpy types (like int64) to standard Python types
        # for JSON conversion to work.
        response = {
            'prediction': int(prediction_value[0]), # e.g., 1
            'confidence': confidence_score           # e.g., 0.88
        }
        
        return jsonify(response)

    except KeyError as e:
        # This error happens if the JSON from the frontend is missing a key
        return jsonify({'error': f'Missing data for feature: {str(e)}'}), 400
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


# --- 4. RUN THE FLASK SERVER ---
# This line only runs if you execute this file directly (e.g., `python app.py`)
if __name__ == '__main__':
    # 'debug=True' means the server will auto-reload if you change the code.
    # Turn this off for a real deployment.
    app.run(port=5000, debug=True)