import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib  # For saving the model
import os      # To check if file exists

print("--- [Part 1] Model Training Script ---")

# --- 1. Define File Names ---
# This is the input file
csv_filename = 'parkinsons.csv'

# These are the output files this script will create
model_filename = 'parkinsons_model.joblib'
scaler_filename = 'data_scaler.joblib'
columns_filename = 'feature_columns.txt'

# --- 2. Load Data ---
if not os.path.exists(csv_filename):
    print(f"Error: The file '{csv_filename}' was not found.")
    print("Please make sure it's in the same directory as this script.")
else:
    try:
        print(f"Loading data from {csv_filename}...")
        df = pd.read_csv(csv_filename)

        # --- 3. Prepare Data for Training ---
        print("Preparing data...")
        
        # 'y' is the target (what we want to predict)
        y = df['status']
        
        # 'X' are the features (the inputs used to make the prediction)
        # We drop 'name' (not a number) and 'status' (the target)
        X = df.drop(['name', 'status'], axis=1)
        
        # Save the exact order of columns for our API
        # This is a *critical* step
        feature_columns = X.columns.tolist()

        # --- 4. Create and "Fit" the Scaler ---
        # We create a scaler to normalize the data
        # .fit_transform() "learns" the scaling rules and applies them
        print("Fitting data scaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- 5. Train the Machine Learning Model ---
        # We'll use a RandomForest, which is powerful and reliable
        print("Training the RandomForest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # .fit() is the "training" step. This is where the model "learns".
        model.fit(X_scaled, y)

        # --- 6. Save the Model and Scaler to Files ---
        # Now we save our "brains" to disk so 'app.py' can use them
        print(f"Saving model to {model_filename}...")
        joblib.dump(model, model_filename)

        print(f"Saving scaler to {scaler_filename}...")
        joblib.dump(scaler, scaler_filename)
        
        print(f"Saving feature columns to {columns_filename}...")
        with open(columns_filename, 'w') as f:
            for col in feature_columns:
                f.write(f"{col}\n")

        print("\n--- SUCCESS! ---")
        print("Your model and scaler have been trained and saved.")
        print("You can now run 'app.py' to start the web server.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your 'parkinsons.csv' file and dependencies.")