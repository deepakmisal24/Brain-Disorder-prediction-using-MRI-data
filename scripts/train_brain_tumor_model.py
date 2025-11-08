import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# --- 1. Define Paths ---
# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The data file is now expected to be in the SAME directory as the script
DATA_PATH = os.path.join(SCRIPT_DIR, 'Brain Tumor.csv')

# Models will be saved in a 'models' directory in the project root (one level up)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

MODEL_PATH = os.path.join(MODELS_DIR, 'brain_tumor_ann.keras')
SCALER_PATH = os.path.join(MODELS_DIR, 'brain_tumor_scaler.joblib')

# The feature names must be in the same order as the training data
FEATURES = [
    'Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness',
    'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity',
    'Dissimilarity', 'Correlation', 'Coarseness'
]

def train_model():
    """
    Loads data, preprocesses, trains the ANN model for brain tumor detection,
    and saves the model and scaler.
    """
    # --- 2. Load Data ---
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_PATH}'.")
        print("Please ensure 'Brain Tumor.csv' is in the 'scripts' directory.")
        return
        
    X = df[FEATURES]
    y = df['Class']

    # --- 3. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # --- 5. Build and Train the ANN Model ---
    model = Sequential([
        Dense(64, activation='relu', input_shape=(len(FEATURES),)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Training the Brain Tumor ANN model...")
    model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # --- 6. Save the Model and Scaler ---
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

if __name__ == '__main__':
    train_model()
