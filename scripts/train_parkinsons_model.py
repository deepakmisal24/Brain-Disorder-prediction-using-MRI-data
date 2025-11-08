import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import os

# --- 1. Define Paths ---
# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The data file is now expected to be in the SAME directory as the script
DATA_PATH = os.path.join(SCRIPT_DIR, 'parkinsons.csv')

# Models will be saved in a 'models' directory in the project root (one level up)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

PIPELINE_PATH = os.path.join(MODELS_DIR, 'parkinsons_svm_pipeline.joblib')

def train_model():
    """
    Loads data, trains the final SVM pipeline on the entire dataset,
    and saves the pipeline.
    """
    # --- 2. Load and Prepare Data ---
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_PATH}'.")
        print("Please ensure 'parkinsons.csv' is in the 'scripts' directory.")
        return
    
    # Drop the non-predictive 'name' column
    if 'name' in df.columns:
        df = df.drop('name', axis=1)

    X = df.drop('status', axis=1)
    y = df['status']
    
    # Store feature names for later use in the prediction app
    feature_names = X.columns.tolist()

    # --- 3. Create and Train the SVM Pipeline ---
    # The pipeline bundles the scaler and the SVM model.
    # probability=True is added to enable confidence scores.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=42, C=10, gamma=0.1, kernel='rbf', probability=True)) # <-- CHANGE IS HERE
    ])

    print("Training the final Tuned SVM Model on the entire dataset (with probabilities)...")
    pipeline.fit(X, y)
    print("Training complete.")

    # --- 4. Save the Entire Pipeline ---
    joblib.dump(pipeline, PIPELINE_PATH)
    
    # Save the feature list so the prediction app knows the column order
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'parkinsons_feature_names.joblib'))

    print(f"Trained pipeline saved to {PIPELINE_PATH}")

if __name__ == '__main__':
    train_model()
