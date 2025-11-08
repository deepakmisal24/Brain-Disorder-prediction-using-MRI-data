import joblib
import os

# --- This script creates the class indices file for the Brain Tumor CNN model ---

def create_indices_file():
    """
    Creates and saves the dictionary that maps class names to integer labels.
    This mapping is based on the alphabetical order of the subdirectories
    in the training dataset.
    """
    print("--- Creating Brain Tumor Class Indices File ---")
    
    # The class names are the folder names from the dataset, sorted alphabetically.
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Create the dictionary mapping: {'glioma': 0, 'meningioma': 1, ...}
    class_indices = {name: i for i, name in enumerate(class_names)}
    
    # Define the output path
    models_dir = 'models'
    output_path = os.path.join(models_dir, 'brain_tumor_cnn_class_indices.joblib')
    
    # Ensure the 'models' directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the dictionary to the file using joblib
    joblib.dump(class_indices, output_path)
    
    print(f"Successfully created class indices file at: '{output_path}'")
    print(f"Contents: {class_indices}")

if __name__ == "__main__":
    create_indices_file()
