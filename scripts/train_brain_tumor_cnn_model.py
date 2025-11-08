import os
import tensorflow as tf
import kagglehub
import warnings
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Import the modern optimizer
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Optimizations for Local CPU/GPU ---
NUM_CPU_THREADS = 4
tf.config.threading.set_inter_op_parallelism_threads(NUM_CPU_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_CPU_THREADS)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- 1. Define Paths and Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

MODEL_PATH = os.path.join(MODELS_DIR, 'brain_tumor_cnn_model.keras')
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, 'brain_tumor_cnn_class_indices.joblib')

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 40
SEED = 111

# --- 2. Create Data Generators ---
def create_data_generators(dataset_path):
    """Creates ImageDataGenerators for training and testing."""
    print("--- Creating Data Generators ---")
    train_dir = os.path.join(dataset_path, 'Training')
    test_dir = os.path.join(dataset_path, 'Testing')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        brightness_range=(0.85, 1.15),
        width_shift_range=0.002,
        height_shift_range=0.002,
        shear_range=12.5,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest"
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED
    )
    
    print(f"Classes found: {list(train_generator.class_indices.keys())}")
    return train_generator, test_generator

# --- 3. Build the Custom CNN Model ---
def build_model(num_classes):
    """Builds the custom CNN model from the notebook."""
    print("--- Building Custom CNN Model ---")
    image_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    model = Sequential([
        Conv2D(32, (4, 4), activation="relu", input_shape=image_shape),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, (4, 4), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (4, 4), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (4, 4), activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5, seed=SEED),
        Dense(num_classes, activation="softmax")
    ])

    # Use the modern Adam optimizer with the same parameters
    optimizer = Adam(learning_rate=0.001, beta_1=0.869, beta_2=0.995)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    return model

# --- 4. Main Training Function ---
def main():
    """Main function to download data, train the model, and save it."""
    print("--- Starting Brain Tumor MRI CNN Model Training (99% Accuracy Notebook) ---")
    
    print("Downloading dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    train_generator, test_generator = create_data_generators(dataset_path)
    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=True, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=True)

    print("\n--- Starting Model Training ---")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    model.save(MODEL_PATH)
    joblib.dump(train_generator.class_indices, CLASS_INDICES_PATH)

    print(f"\n--- Training Complete ---")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class indices saved to: {CLASS_INDICES_PATH}")

if __name__ == '__main__':
    main()
