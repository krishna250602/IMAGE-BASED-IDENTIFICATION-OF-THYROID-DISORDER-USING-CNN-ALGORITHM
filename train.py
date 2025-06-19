from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os

# Constants
IMAGE_SIZE = (30, 30)  # Image dimensions
NUM_CLASSES = 6  # Set to 5 since you have 5 classes

# Data preparation
data = []
labels = []
cur_path = os.getcwd()  # Get current directory

class_labels = {
    0: "hyperthyroidism_ultrasound_images",
    1: "hypothyroidism_ultrasound_images",
    2: "thyroid_cancer_ultrasound_images",
    3: "thyroid_nodule_ultrasound_images",
    4: "thyroid_normal_ultrasound_images",
    5: "non_thyroid_images"
   
}

# Retrieve the images and their labels
print("Obtaining Images & their Labels..............")
for label, folder_name in class_labels.items():
    path = os.path.join(cur_path, 'Dataset/train/', folder_name)
    images = os.listdir(path)
    
    for img_name in images:
        try:
            image_path = os.path.join(path, img_name)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize(IMAGE_SIZE)  # Resize to 30x30
            image = np.array(image)  # Convert to numpy array
            image = np.expand_dims(image, axis=-1)  # Add channel dimension (1 for grayscale)
            image = image / 255.0  # Normalize the image to [0, 1]
            data.append(image)
            labels.append(label)  # Append the label index, not the folder name
            print(f"{img_name} Loaded")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")

print("Dataset Loaded")

# Convert lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

# Split training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Convert the labels into one-hot encoding
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print("Training under process...")
model = Sequential([
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),  # Input shape (30, 30, 1) for grayscale
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(rate=0.5),
    Dense(NUM_CLASSES, activation='softmax')  # Output layer for multi-class classification
])

print("Initialized model")

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Save the trained model
model.save("my_model_cnn.h5")

print("Model training complete and saved.")
