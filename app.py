# Import libraries and sub files
import streamlit as st
from PIL import Image
from utils import (resize_image_for_display, apply_lesion_on_white_background,
                   apply_black_lesion_on_white_background, load_and_prep_image_segmentation,
                   preprocess_for_subtype, predict_subtype, resize_mask_for_display,
                   load_preprocess_image_classification, subtype_full_names,
                   get_img_array, make_gradcam_heatmap, display_gradcam)
from models import (load_segmentation_model, load_subtype_model, 
                    load_classification_model, predict_with_model)
import numpy as np
import cv2 
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load models
classification_model = load_classification_model()
subtype_model = load_subtype_model()
segmentation_model = load_segmentation_model()

# Define the classes list in the app.py file
classes = ['Benign', 'Malignant', 'Normal']

img_size = (256, 256) 

# Streamlit application layout
st.title('Breast Cancer Analysis Tool')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Resize for display with new function
    display_image = resize_image_for_display(image, max_display_size=(300, 300))
    
    # Display the resized image
    st.image(display_image, caption='Uploaded Image', use_column_width=False)
    
    # ---------------------Classification---------------------
    processed_image_classification = load_preprocess_image_classification(uploaded_file)
    prediction = predict_with_model(classification_model, processed_image_classification)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Extract the first element
    predicted_class_name = classes[predicted_class_index]
    st.write(f'Classification Prediction: {predicted_class_name}')

    # ---------------------Subtype prediction, Grad CAM and segmentation---------------------
    is_malignant = predicted_class_name == 'Malignant'
    if predicted_class_name in ['Malignant', 'Benign']:
        # ---------------------Diagnosis classification---------------------
        subtype_full_name = predict_subtype(subtype_model, image, is_malignant)
        st.write(f'Diagnosis Prediction: {subtype_full_name}')
    
        # ---------------------Applying Grad CAM---------------------
        # Preprocess the image for Grad-CAM
        gradcam_img_array = preprocess_input(get_img_array(uploaded_file, img_size))

        # Generate heatmap
        heatmap = make_gradcam_heatmap(gradcam_img_array, classification_model, 'conv5_block3_out')

        # Display Grad-CAM heatmap
        st.write("Grad-CAM Heatmap:")
        gradcam_display_size = (300, 300)  # Set the desired display size
        # ... (after generating the heatmap)
        gradcam_image = display_gradcam(uploaded_file, heatmap, display_size=gradcam_display_size)
        st.image(gradcam_image, caption='Grad-CAM Heatmap', use_column_width=False)  # Set use_column_width to False

        # ---------------------U-Net Segmentation---------------------
        st.write("Proceeding to segmentation...")
        processed_image_segmentation = load_and_prep_image_segmentation(uploaded_file)
        pred_mask = predict_with_model(segmentation_model, processed_image_segmentation)

        # Remove the batch and channel dimensions from the binary mask
        binary_mask = np.squeeze((pred_mask > 0.5).astype(np.uint8))

        original_image_np = np.array(image)

        # Check if the dimensions are correct before resizing
        if binary_mask.ndim != 2:
            st.error("The binary mask has an unexpected number of dimensions.")
            st.stop()

        # Now that we've confirmed the binary_mask is 2D, we can resize it
        binary_mask_resized = cv2.resize(binary_mask, (original_image_np.shape[1], original_image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Task selection
        task_option = st.selectbox("Select the task:", 
                                   ("Original Lesion", 
                                    "Mask Lesion"))
        
        # Process the binary mask and prepare the images for each task
        if task_option == "Original Lesion":
            processed_image = apply_lesion_on_white_background(original_image_np, binary_mask_resized)
        elif task_option == "Mask Lesion":
            processed_image = apply_black_lesion_on_white_background(original_image_np, binary_mask_resized)

        # Convert the processed image back to PIL Image for display in Streamlit
        processed_image_pil = Image.fromarray(processed_image)

        # Resize the processed image for display
        display_processed_image = resize_image_for_display(processed_image_pil)

        # Display the original image and the processed image according to the selected task
        col1, col2 = st.columns(2)
        with col1:
            st.image(display_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(display_processed_image, caption=f"Processed Image ({task_option})", use_column_width=True)