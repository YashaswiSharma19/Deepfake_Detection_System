import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define paths to the dataset
train_data_dir = "C:\\Users\\LENOVO\\Downloads\\img_data-20240915T141034Z-001\\img_data\\train_img" #path to train data 
test_data_dir = "C:\\Users\\LENOVO\\Downloads\\img_data-20240915T141034Z-001\\img_data\\test_img" #path to test data
train_datagen = ImageDataGenerator(
    rescale=1./255, # Example preprocessing - normalize pixel values
    # Add other data augmentation options here if needed
)

test_datagen = ImageDataGenerator(rescale=1./255)
# Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,             # Path to train dataset
    target_size=(224,224),       # Resizing the images
    batch_size=32,
    class_mode='binary',   # For multi-class classification
)
# Load test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,              # Path to test dataset
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',   # Use 'binary' for binary classification
)
# Check the class labels (optional)
print("Class Indices:", train_generator.class_indices)
import tensorflow
from tensorflow import keras
from keras.layers import Dense,GlobalAveragePooling2D,Flatten
from keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications import ResNet50
#setting the conv layers as that of vgg16
conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
model=keras.models.Sequential()
model.add(conv_base)
#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

from tensorflow.keras.models import Sequential
model = Sequential([conv_base,
                    GlobalAveragePooling2D(),
                    Dense(256, activation="relu"),
                    Dense(1, activation="sigmoid")])

conv_base.trainable=False
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,epochs=10,validation_data=test_generator)
