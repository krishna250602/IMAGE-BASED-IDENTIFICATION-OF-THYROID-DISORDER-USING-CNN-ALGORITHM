import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Class mapping
class_labels = {
    0: "Hyperthyroidism",
    1: "Hypothyroidism",
    2: "Thyroid Cancer",
    3: "Thyroid Nodule",
    4: "Thyroid Normal",
    5: "non_thyroid_images"
    
    
}

# Load the trained model
model_path = 'my_model_cnn.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
model = load_model(model_path, compile=False)
print("Loaded model from disk")

# Function to preprocess and classify an image
def classify(img_path):
    try:
        # Load and preprocess the image
        test_image = Image.open(img_path).convert('L')  # Convert to grayscale
        test_image = test_image.resize((30, 30))  # Resize to match model's input size
        test_image = np.array(test_image) / 255.0  # Normalize to [0, 1]
        test_image = np.expand_dims(test_image, axis=-1)  # Add channel dimension
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Output prediction and confidence
        print(f"Image Path: {img_path}")
        print(f"Prediction: {class_labels.get(predicted_class, 'Unknown')} ({confidence:.2f} confidence)")

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

# Test dataset path
# hyperthyroidism_ultrasound_images
# hypothyroidism_ultrasound_images
# thyroid_cancer_ultrasound_images
# thyroid_nodule_ultrasound_images
# thyroid_normal_ultrasound_images
# non_thyroid_images
test_path = os.path.join(os.getcwd(), 'Dataset/train/hyperthyroidism_ultrasound_images/')
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test dataset path '{test_path}' does not exist.")

# Find all test images
test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith(('.jpeg', '.jpg'))]

# Ensure there are files to classify
if not test_images:
    print("No .jpeg or .jpg files found in the test dataset.")
else:
    # Classify each test image
    for img_file in test_images:
        classify(img_file)
        print('\n')
