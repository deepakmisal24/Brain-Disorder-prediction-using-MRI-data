from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# --- Helper function to load models safely ---
def load_prediction_model(model_name):
    """Loads a model from the 'models' directory."""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'models', model_name)
        if '.keras' in model_name:
            # compile=False can speed up loading for inference-only models
            return load_model(model_path, compile=False)
        else:
            return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

# --- Page Rendering Views ---

def index(request):
    """Renders the main landing page."""
    return render(request, 'predictor/index.html')

# --- Alzheimer's Prediction View ---
def alzheimers_page(request):
    if request.method == 'POST':
        if 'mri_image' in request.FILES:
            image_file = request.FILES['mri_image']
            model = load_prediction_model('alzheimers_cnn_model.keras')
            if model is None: return JsonResponse({'error': 'Alzheimer\'s CNN model not found.'}, status=500)

            img = Image.open(image_file).resize((128, 128))
            img_array = np.array(img)
            if img_array.ndim == 2: img_array = np.stack([img_array]*3, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            class_names = {0: 'Non Demented', 1: 'Mild Dementia', 2: 'Moderate Dementia', 3: 'Very Mild Dementia'}
            prediction = class_names.get(pred_index, "Unknown")
            return JsonResponse({'prediction': prediction, 'confidence': f'{confidence:.2f}%'})
        else:
            model = load_prediction_model('alzheimers_gb_model.joblib')
            scaler = load_prediction_model('alzheimers_scaler.joblib')
            if model is None or scaler is None: return JsonResponse({'error': 'Alzheimer\'s numerical model not found.'}, status=500)

            try:
                input_data = [
                    float(request.POST.get('m_f')), float(request.POST.get('age')),
                    float(request.POST.get('educ')), float(request.POST.get('ses')),
                    float(request.POST.get('mmse')), float(request.POST.get('etiv')),
                    float(request.POST.get('nwbv')), float(request.POST.get('asf')),
                ]
            except (ValueError, TypeError):
                return JsonResponse({'error': 'Invalid or missing numerical input.'}, status=400)

            feature_order = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
            input_df = pd.DataFrame([input_data], columns=feature_order)
            input_scaled = scaler.transform(input_df)

            pred_proba = model.predict_proba(input_scaled)
            pred_index = np.argmax(pred_proba[0])
            confidence = np.max(pred_proba[0]) * 100
            class_names = {0: 'Healthy', 1: 'Very Mild Dementia', 2: 'Mild Dementia', 3: 'Moderate Dementia'}
            prediction = class_names.get(pred_index, "Unknown")
            return JsonResponse({'prediction': prediction, 'confidence': f'{confidence:.2f}%'})

    return render(request, 'predictor/alzheimers.html')

# --- Brain Tumor Prediction View ---
def braintumor_page(request):
    if request.method == 'POST':
        if 'mri_image' in request.FILES:
            image_file = request.FILES['mri_image']
            model = load_prediction_model('brain_tumor_cnn_model.keras')
            class_indices = load_prediction_model('brain_tumor_cnn_class_indices.joblib')
            if model is None or class_indices is None: return JsonResponse({'error': 'Brain Tumor CNN model not found.'}, status=500)

            # --- CORRECTED IMAGE SIZE ---
            img = Image.open(image_file).resize((150, 150)) # Changed from (299, 299)
            img_array = np.array(img)
            if img_array.ndim == 2: img_array = np.stack([img_array]*3, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            
            # The new model expects pixel values to be rescaled, which is done in the model itself.
            # No need for img_array = img_array / 255.0 here.

            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            class_names = {v: k for k, v in class_indices.items()}
            prediction = class_names.get(pred_index, "Unknown").capitalize()
            return JsonResponse({'prediction': prediction, 'confidence': f'{confidence:.2f}%'})
        else:
            model = load_prediction_model('brain_tumor_ann.keras')
            scaler = load_prediction_model('brain_tumor_scaler.joblib')
            if model is None or scaler is None: return JsonResponse({'error': 'Brain Tumor ANN model not found.'}, status=500)

            try:
                input_data = [
                    float(request.POST.get('mean')), float(request.POST.get('variance')),
                    float(request.POST.get('std_dev')), float(request.POST.get('entropy')),
                    float(request.POST.get('skewness')), float(request.POST.get('kurtosis')),
                    float(request.POST.get('contrast')), float(request.POST.get('energy')),
                    float(request.POST.get('asm')), float(request.POST.get('homogeneity')),
                    float(request.POST.get('dissimilarity')), float(request.POST.get('correlation')),
                    float(request.POST.get('coarseness')),
                ]
            except (ValueError, TypeError):
                 return JsonResponse({'error': 'Invalid or missing numerical input.'}, status=400)

            feature_order = ['Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Coarseness']
            input_df = pd.DataFrame([input_data], columns=feature_order)
            input_scaled = scaler.transform(input_df)
            pred_proba = model.predict(input_scaled)[0][0]
            prediction = "Tumor Detected" if pred_proba > 0.5 else "No Tumor"
            confidence = (pred_proba if pred_proba > 0.5 else 1 - pred_proba) * 100
            return JsonResponse({'prediction': prediction, 'confidence': f'{confidence:.2f}%'})

    return render(request, 'predictor/braintumor.html')

# --- Parkinson's Prediction View ---
def parkinsons_page(request):
    if request.method == 'POST':
        if 'mri_image' in request.FILES:
            image_file = request.FILES['mri_image']
            model = load_prediction_model('parkinsons_cnn_model.keras')
            if model is None: return JsonResponse({'error': 'Parkinson\'s CNN model not found.'}, status=500)

            img = Image.open(image_file).resize((224, 224))
            img_array = np.array(img)
            if img_array.ndim == 2: img_array = np.stack([img_array]*3, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)

            pred_proba = model.predict(img_array)[0][0]
            prediction = "Parkinson's Detected" if pred_proba > 0.5 else "Healthy"
            confidence = (pred_proba if pred_proba > 0.5 else 1 - pred_proba) * 100
            return JsonResponse({'prediction': prediction, 'confidence': f'{confidence:.2f}%'})
        else:
            pipeline = load_prediction_model('parkinsons_svm_pipeline.joblib')
            feature_names = load_prediction_model('parkinsons_feature_names.joblib')
            if pipeline is None or feature_names is None: return JsonResponse({'error': 'Parkinson\'s numerical model not found.'}, status=500)

            try:
                input_data = [float(request.POST.get(f)) for f in feature_names]
            except (ValueError, TypeError):
                return JsonResponse({'error': 'Invalid numerical input.'}, status=400)

            input_df = pd.DataFrame([input_data], columns=feature_names)

            pred_proba = pipeline.predict_proba(input_df)
            prediction = "Parkinson's Detected" if pred_proba[0][1] > 0.5 else "Healthy"
            confidence = np.max(pred_proba[0]) * 100
            return JsonResponse({'prediction': prediction, 'confidence': f'{confidence:.2f}%'})

    return render(request, 'predictor/parkinsons.html')
