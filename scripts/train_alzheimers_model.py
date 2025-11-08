import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- 1. Define Paths ---
# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The data file is now expected to be in the SAME directory as the script
DATA_PATH = os.path.join(SCRIPT_DIR, 'oasis_cross-sectional.xlsx')

# Models will be saved in a 'models' directory in the project root (one level up)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

MODEL_PATH = os.path.join(MODELS_DIR, 'alzheimers_gb_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'alzheimers_scaler.joblib')

# --- 2. Load and Preprocess Data ---
def train_model():
    """Loads data, preprocesses, trains the model, and saves it."""
    try:
        df = pd.read_excel(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_PATH}'.")
        print("Please ensure 'oasis_cross-sectional.xlsx' is in the 'scripts' directory.")
        return

    # Preprocessing steps
    df_clean = df.dropna(subset=['CDR', 'MMSE', 'Educ']).copy()
    median_ses = df_clean['SES'].median()
    df_clean.loc[:, 'SES'] = df_clean['SES'].fillna(median_ses)
    
    columns_to_drop = ['ID', 'Hand', 'Delay']
    columns_to_drop_exist = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns_to_drop_exist, axis=1)

    df_clean.loc[:, 'M/F'] = df_clean['M/F'].apply(lambda x: 1 if x == 'M' else 0)

    cdr_mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
    df_clean.loc[:, 'CDR_Class'] = df_clean['CDR'].map(cdr_mapping)

    X = df_clean.drop(['CDR', 'CDR_Class'], axis=1)
    y = df_clean['CDR_Class']
    
    feature_order = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    missing_features = [col for col in feature_order if col not in X.columns]
    if missing_features:
        print(f"Error: The following required features are missing from the data: {missing_features}")
        return
        
    X = X[feature_order]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Scale and Train ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    gb_model = GradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=200,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)

    # --- 4. Save Model and Scaler ---
    joblib.dump(gb_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

if __name__ == '__main__':
    train_model()
