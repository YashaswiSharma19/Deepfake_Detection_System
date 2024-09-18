import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from PIL import Image

# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("C:\\Users\\LENOVO\\Downloads\\deepfake_detection_model.h5") #path to the trained model saved as .h5 file
    return model

# Function to preprocess and predict whether the image is real or fake
def preprocess_and_predict(image, model):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(img_array)
    return prediction[0][0]

# Streamlit Frontend
st.title("Deepfake Detection System")
st.write("Upload an image to check if it's a deepfake or real.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load the model
model = load_model()

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing the image and make predictions
    st.write("Classifying...")
    prediction = preprocess_and_predict(img, model)

    # Displaying the result with a more informative and styled output
    if prediction > 0.5:
        st.markdown(f"<h3 style='color:green;'>The image is classified as: <strong>Real</strong> with {prediction*100:.2f}% confidence.</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red;'>The image is classified as: <strong>Fake</strong> with {(1-prediction)*100:.2f}% confidence.</h3>", unsafe_allow_html=True)

    # Adding additional information or instructions
    st.write("Ensure the image is clear and well-lit for accurate results.")
    st.write("If the result is unexpected, try uploading another image or check the model performance.")

# Adding custom styling for the page
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f5;
    }
    </style>
""", unsafe_allow_html=True)

