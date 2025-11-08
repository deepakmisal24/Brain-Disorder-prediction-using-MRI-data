import os
import numpy as np
import pandas as pd
import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
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

MODEL_PATH = os.path.join(MODELS_DIR, 'parkinsons_cnn_model.keras')
IMAGE_SIZE = (224, 224)
EPOCHS = 25
BATCH_SIZE = 32

# --- 2. Load and Prepare Data ---
def load_and_prepare_data(dataset_path):
    """Scans the dataset directory, creates a DataFrame, and splits it."""
    print("Loading and preparing data...")
    data_dir = Path(dataset_path)
    
    # Create a DataFrame with paths and labels
    df = pd.DataFrame({'path': list(data_dir.glob('**/*.png'))})
    df['disease'] = df['path'].map(lambda x: x.parent.stem)
    df['path'] = df['path'].astype(str) # Convert Path objects to strings for the generator

    # Split into training and testing sets
    X = df['path']
    y = df['disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_df = pd.DataFrame({'path': X_train, 'disease': y_train})
    test_df = pd.DataFrame({'path': X_test, 'disease': y_test})
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    return train_df, test_df

# --- 3. Create Data Generators ---
def create_data_generators(train_df, test_df):
    """Creates ImageDataGenerators for training and testing."""
    print("Creating data generators...")
    # Use data augmentation for the training set to improve model robustness
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescale the test set
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='disease',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='disease',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False # Keep order for evaluation
    )
    return train_generator, test_generator

# --- 4. Build the Model (Transfer Learning) ---
def build_model():
    """Builds a transfer learning model using ResNet50."""
    print("Building model with ResNet50 base...")
    # Load ResNet50 with pre-trained ImageNet weights, without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Freeze the layers of the base model so they are not trained
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom classification head
    top_model = base_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dropout(0.5)(top_model)
    # The final Dense layer with a sigmoid activation gives the prediction and confidence score
    output_layer = Dense(1, activation='sigmoid')(top_model)
    
    model = Model(inputs=base_model.input, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 5. Main Training Function ---
def main():
    """Main function to download data, train the model, and save it."""
    print("--- Starting Parkinson's MRI CNN Model Training ---")
    
    # Download the dataset
    print("Downloading dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("irfansheriff/parkinsons-brain-mri-dataset")
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have authenticated with Kaggle (e.g., via kaggle.json).")
        return

    # Prepare data and generators
    train_df, test_df = load_and_prepare_data(dataset_path)
    train_generator, test_generator = create_data_generators(train_df, test_df)

    # Build and train the model
    model = build_model()
    print("\nModel Summary:")
    model.summary()
    
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        batch_size=BATCH_SIZE
    )

    # Save the final model
    model.save(MODEL_PATH)
    print(f"\n--- Training Complete ---")
    print(f"Parkinson's CNN model saved to: {MODEL_PATH}")

if __name__ == '__main__':
    main()
