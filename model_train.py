import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np

# Paths to the dataset
train_data_dir = '/content/drive/MyDrive/img_data/train_img'  # Adjust these paths accordingly
test_data_dir = '/content/drive/MyDrive/img_data/test_img'

# Function to preprocess images: Grayscale to RGB conversion
def preprocess_image(img):
    if len(img.shape) == 2:  # If the image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    elif img.shape[2] == 1:  # If the image has a single channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

# Custom function to apply preprocessing to all images
def preprocess_input(img):
    img = preprocess_image(img)  # Apply grayscale to RGB conversion if needed
    return img / 255.0  # Normalize pixel values between 0 and 1

# Image data generators with preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Custom preprocessing function
    shear_range=0.2,             # Data augmentation: shearing
    zoom_range=0.2,              # Data augmentation: zooming
    horizontal_flip=True         # Data augmentation: flipping
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Apply preprocessing to test data

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',          # Binary classification (real vs fake)
    shuffle=True
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',          # Binary classification
    shuffle=False
)

# Load MobileNet, without the top fully connected layers
conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the model
model = Sequential([
    conv_base,
    GlobalAveragePooling2D(),     # Better for deep models than Flatten
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Freeze the convolutional base
conv_base.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Model evaluation on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

#saving the model 
model.save('deepfake_detection_model.h5')

#downloading the file as .h5 file
from google.colab import files
files.download('deepfake_detection_model.h5')

print(train_generator.class_indices)
