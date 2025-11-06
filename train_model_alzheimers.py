import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import numpy as np
import warnings

# --- Configuration ---
warnings.filterwarnings('ignore')
DATASET_PATH = 'oasis_longitudinal.csv'
MODEL_FILENAME = 'alzheimers_model.joblib'
SCALER_FILENAME = 'alzheimers_scaler.joblib'
SES_IMPUTER_FILENAME = 'ses_imputer.joblib'
MMSE_IMPUTER_FILENAME = 'mmse_imputer.joblib'

print(f"[INFO] Starting Alzheimer's model training using {DATASET_PATH}...")

# --- 1. Load Data ---
df = pd.read_csv(DATASET_PATH)

# --- 2. Data Preprocessing & Feature Engineering ---
# Only use the *first visit* for each subject (Visit = 1)
# This prevents data leakage and models a "first-time diagnosis".
df_first_visit = df[df['Visit'] == 1].copy()


# --- Define Target (y) ---
df_first_visit['Group'] = df_first_visit['Group'].replace({'Converted': 'Nondemented'})
df_first_visit['y'] = df_first_visit['Group'].map({'Nondemented': 0, 'Demented': 1})
df_first_visit = df_first_visit.dropna(subset=['y'])
y = df_first_visit['y']

# --- Define Features (X) ---
FEATURES = [
    'M/F',   # Gender
    'Age',   # Age
    'EDUC',  # Years of Education
    'SES',   # Socioeconomic Status (has NaNs)
    'MMSE',  # Mini-Mental State Exam (has NaNs)
    'eTIV',  # Estimated Total Intracranial Volume
    'nWBV',  # Normalized Whole Brain Volume
    'ASF'    # Atlas Scaling Factor
]
X = df_first_visit[FEATURES].copy()

# --- 3. Handle Missing Data (Imputation) ---
# We must save these imputers to use in the API
ses_imputer = SimpleImputer(strategy='median')
X['SES'] = ses_imputer.fit_transform(X[['SES']])

# Impute 'MMSE'
mmse_imputer = SimpleImputer(strategy='median')
X['MMSE'] = mmse_imputer.fit_transform(X[['MMSE']])

# --- 4. Handle Categorical Data ---
X['M/F'] = X['M/F'].map({'M': 0, 'F': 1})

# --- 5. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# --- 6. Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 7. Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# --- 8. Evaluate Model ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# --- 9. Save Model, Scaler, and Imputers ---
joblib.dump(model, MODEL_FILENAME)
joblib.dump(scaler, SCALER_FILENAME)
joblib.dump(ses_imputer, SES_IMPUTER_FILENAME)
joblib.dump(mmse_imputer, MMSE_IMPUTER_FILENAME)
print(f"[SUCCESS] All model files saved.")