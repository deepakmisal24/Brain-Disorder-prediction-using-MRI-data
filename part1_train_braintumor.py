# -----------------------------------------------------------------
# part1_train_braintumor.py
# -----------------------------------------------------------------
# This script trains a classifier on the numerical (tabular)
# brain tumor dataset.
#
# It does the following:
# 1. Loads the data from 'Brain Tumor(2).csv'.
# 2. Prepares the data by separating features (X) and target (y).
# 3. Splits the data into training and testing sets.
# 4. Applies feature scaling (StandardScaler).
# 5. Trains a RandomForestClassifier model.
# 6. Evaluates the model on the test set.
# 7. Saves the trained model ('brain_tumor_model.joblib')
#    and the scaler ('scaler.joblib') for use in an API.
# -----------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

# Suppress any future warnings
warnings.filterwarnings('ignore')

print("[INFO] Script starting...")

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
    
# 'Image' is an identifier, not a feature. We drop it.
# 'Class' is our target variable (y).
# Everything else is a feature (X).
try:
    X = df.drop(columns=['Image', 'Class'])
    y = df['Class']
    print("[INFO] Features (X) and Target (y) created.")
    print(f"[INFO] {len(X.columns)} feature columns: {list(X.columns)}")
except KeyError:
    # This might happen if 'Image' isn't in the file
    X = df.drop(columns=['Class'])
    y = df['Class']
    print("[INFO] 'Image' column not found, proceeding with all other columns as features.")
    print(f"[INFO] {len(X.columns)} feature columns: {list(X.columns)}")


# --- 3. Split Data ---
# We split into 80% training and 20% testing.
# 'stratify=y' ensures that the train and test sets have the
# same proportion of 0s and 1s as the original dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,     # 20% for testing
    random_state=42,  # For reproducible results
    stratify=y        # Keep class balance
)
print(f"[INFO] Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# --- 4. Scale Features ---
# We use StandardScaler to give all features a mean of 0 and std dev of 1.
# This is crucial for many models and generally good practice.
scaler = StandardScaler()

# CRITICAL: We 'fit' the scaler ONLY on the training data.
scaler.fit(X_train)

# Then we 'transform' both the training and test data.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[INFO] Features scaled successfully.")

# --- 5. Train Model ---
# RandomForest is a powerful and robust model for tabular data.
# n_jobs=-1 uses all available CPU cores to speed up training.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("[INFO] Training RandomForestClassifier...")
model.fit(X_train_scaled, y_train)
print("[INFO] Model training complete.")

# --- 6. Evaluate Model ---
# Let's see how the model performs on the unseen test data.
print("\n" + "="*30)
print("--- Model Evaluation on Test Data ---")
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
# [[True Neg (0,0), False Pos (0,1)],
#  [False Neg (1,0), True Pos (1,1)]]
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0 (No Tumor)', 'Class 1 (Tumor)']))
print("="*30)

# --- 7. Save Model and Scaler ---
# These two files are the final product of this script.
model_filename = 'brain_tumor_model.joblib'
scaler_filename = 'brain_tumor_scaler.joblib'

joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"\n[SUCCESS] Model saved to: {model_filename}")
print(f"[SUCCESS] Scaler saved to: {scaler_filename}")
print("[INFO] Script finished.")