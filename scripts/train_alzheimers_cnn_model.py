import os
import numpy as np
import kagglehub
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- 1. Define Paths and Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

MODEL_PATH = os.path.join(MODELS_DIR, 'alzheimers_cnn_model.keras')
IMAGE_SIZE = (128, 128)
# Using a smaller sample size as in the notebook to speed up training
SAMPLE_SIZE = 100 

# --- 2. Download and Load Data ---
def load_and_preprocess_data(dataset_path):
    """Loads images from the dataset path, resizes them, and one-hot encodes the labels."""
    print("Loading and preprocessing image data...")
    
    # Define the subdirectories for each class
    base_data_path = os.path.join(dataset_path, 'Data')
    class_names = ['Non Demented', 'Mild Dementia', 'Moderate Dementia', 'Very mild Dementia']
    
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array(range(len(class_names))).reshape(-1, 1))

    data = []
    result = []

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(base_data_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: Directory not found for class '{class_name}' at {class_path}")
            continue
            
        filenames = os.listdir(class_path)[:SAMPLE_SIZE]
        for filename in filenames:
            try:
                img_path = os.path.join(class_path, filename)
                img = Image.open(img_path)
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img)
                
                # Ensure the image is in RGB format
                if img_array.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3):
                    data.append(img_array)
                    result.append(encoder.transform([[i]]))
            except Exception as e:
                print(f"Could not process image {img_path}: {e}")

    if not data:
        print("Error: No data was loaded. Please check the dataset path and contents.")
        return None, None

    data = np.array(data)
    result = np.array(result).reshape(-1, len(class_names))
    
    print(f"Loaded {data.shape[0]} images.")
    return data, result

# --- 3. Build the CNN Model ---
def build_cnn_model(input_shape, num_classes):
    """Builds the Convolutional Neural Network model."""
    model = Sequential([
        Conv2D(32, kernel_size=(2, 2), input_shape=input_shape, padding='Same'),
        Conv2D(32, kernel_size=(2, 2), activation='relu', padding='Same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(2, 2), activation='relu', padding='Same'),
        Conv2D(64, kernel_size=(2, 2), activation='relu', padding='Same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    return model

# --- 4. Main Training Function ---
def main():
    """Main function to download data, train the model, and save it."""
    print("--- Starting Alzheimer's CNN Model Training ---")
    
    # Download the dataset from Kaggle
    print("Downloading dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("ninadaithal/imagesoasis")
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have authenticated with Kaggle (e.g., via kaggle.json).")
        return

    # Load and preprocess the data
    data, result = load_and_preprocess_data(dataset_path)
    if data is None:
        return

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, result, test_size=0.15, shuffle=True, random_state=42
    )

    # Build the model
    num_classes = y_train.shape[1]
    model = build_cnn_model(x_train.shape[1:], num_classes)
    print("\nModel Summary:")
    model.summary()

    # Train the model
    print("\nStarting model training...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=10,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"\n--- Training Complete ---")
    print(f"CNN model saved to: {MODEL_PATH}")

if __name__ == '__main__':
    main()
