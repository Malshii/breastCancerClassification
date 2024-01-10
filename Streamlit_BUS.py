import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Function to load and preprocess the image
def load_preprocess_image(uploaded_file):
    SIZE = 224  # match the model's expected input size

    # Read and resize the image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((SIZE, SIZE))

    # Apply custom preprocessing if required
    processed_image = custom_preprocessing(np.array(image))

    # Reshape for model input
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Function to add noise, blur, and adjust contrast/brightness
def custom_preprocessing(image):
    # Apply your custom preprocessing steps here
    # For example: Noise addition, blurring, and contrast adjustment
    return image  # Return the processed image

# Function to predict the class
def predict(image):
    # Load the model (adjust the path to your model)
    model = load_model('breast_cancer_model.h5')
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)

# Streamlit application layout
st.title('Breast Cancer Classification')
st.write('This tool classifies breast cancer images into benign, malignant, or normal classes.')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Process and predict
    processed_image = load_preprocess_image(uploaded_file)
    prediction = predict(processed_image)

    # Display prediction
    classes = ['Benign', 'Malignant', 'Normal']
    st.write(f'Prediction: {classes[prediction[0]]}')

# Optional: Display model summary, dataset visualization, etc.