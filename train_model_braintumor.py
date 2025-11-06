
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

# Suppress any future warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
try:
    df = pd.read_csv("Brain Tumor(2).csv")
    print("[INFO] Data loaded successfully.")
except FileNotFoundError:
    print("[ERROR] 'Brain Tumor(2).csv' not found. Please check the file path.")
    exit()

# --- 2. Prepare Data (Define X and y) ---
# Check if required columns exist
if 'Class' not in df.columns:
    print("[ERROR] Target column 'Class' not found in CSV.")
    exit()

# 'Class' is our target variable (y).
# Everything else is a feature (X).
try:
    X = df.drop(columns=['Image', 'Class'])
    y = df['Class']
except KeyError:
    # This might happen if 'Image' isn't in the file
    X = df.drop(columns=['Class'])
    y = df['Class']


# --- 3. Split Data ---
# We split into 80% training and 20% testing..
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,
    random_state=42,  
    stratify=y        
)

# --- 4. Scale Features ---
# We use StandardScaler to give all features a mean of 0 and std dev of 1.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# --- 6. Evaluate Model ---
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# --- 7. Save Model and Scaler ---
model_filename = 'brain_tumor_model.joblib'
scaler_filename = 'brain_tumor_scaler.joblib'

joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

print("[INFO] Script finished.")