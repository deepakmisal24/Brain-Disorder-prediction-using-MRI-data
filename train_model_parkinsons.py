
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

DATASET_PATH = 'parkinsons.csv' # Path to the Parkinson's dataset

FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
] #feature columns to use

# The target we are trying to predict
TARGET = 'status'

print("[INFO] Script starting...")

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] Data loaded successfully from {DATASET_PATH}.")
except FileNotFoundError:
    print(f"[ERROR] '{DATASET_PATH}' not found. Please check the file path.")
    exit()

# --- 2. Prepare Data (Define X and y) ---
try:
    # X = The feature columns
    X = df[FEATURES]
    # y = The 'status' column (0 or 1)
    y = df[TARGET]
    
    print(f"[INFO] Features (X) and Target (y) created.")

except KeyError as e:
    print(f"[ERROR] A required column is missing: {e}")
    exit()

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --- 4. Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# --- 6. Evaluate Model ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# --- 7. Save Model and Scaler ---
# This will OVERWRITE your old files.
model_filename = 'parkinsons_model.joblib'
scaler_filename = 'parkinsons_scaler.joblib'

joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

print("[INFO] Script finished.")