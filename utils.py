# Import libraries
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import backend as K
import cv2

# ---------------------Classification---------------------

# Function to load and preprocess the image for classification
def load_preprocess_image_classification(uploaded_file, size=256):
    # Read and resize the image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((size, size))
    processed_image = np.array(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def resize_image_for_display(image, max_display_size=(300, 300)):
    """
    Resize the uploaded image for display on Streamlit while maintaining the aspect ratio.
    This does not affect the image used for model predictions.
    """
    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        # Image is wider than it is tall
        new_width = min(image.width, max_display_size[0])
        new_height = int(new_width / aspect_ratio)
    else:
        # Image is taller than it is wide
        new_height = min(image.height, max_display_size[1])
        new_width = int(new_height * aspect_ratio)
        
    return image.resize((new_width, new_height), Image.LANCZOS)

# ---------------------U-Net Segmentation---------------------

# Define dice_coefficient
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

# Define precision
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + K.epsilon())
    return precision_val

# Define recall
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positives / (possible_positives + K.epsilon())
    return recall_val

# Define dice_loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Define weighted_custom_loss
def weighted_custom_loss(weight_factor):
    def custom_loss(y_true, y_pred):
        # Counting the number of white pixels (tumor region) in the true mask
        tumor_size = tf.reduce_sum(tf.cast(tf.equal(y_true, 255), tf.float32))

        # Calculate weights based on tumor size - larger tumors get higher weights
        weights = 1 + weight_factor * tumor_size

        # Standard binary cross-entropy loss
        binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)

        # Apply weights to the loss
        weighted_loss = weights * binary_cross_entropy_loss

        # Return the mean loss
        return tf.reduce_mean(weighted_loss)
    return custom_loss

# Assuming the weight_factor was 0.5 when the model was trained
weight_factor = 0.5
custom_loss_fn = weighted_custom_loss(weight_factor)

def load_and_prep_image_segmentation(uploaded_file, img_shape=128):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((img_shape, img_shape))
    img = np.array(img) / 255.0
    return img

def predict_mask(model, img):
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    pred_mask = model.predict(img_array)
    return np.squeeze(pred_mask)  # Remove batch dimension for further processing

def apply_lesion_on_white_background(original_image, binary_mask):
    # Convert binary mask to boolean
    binary_mask_boolean = binary_mask.astype(bool)
    
    # Prepare a white background
    white_background = np.ones_like(original_image) * 255
    
    # Use the binary mask to select the lesion and white background
    result_image = np.where(binary_mask_boolean[..., None], original_image, white_background)
    
    return result_image

def apply_black_lesion_on_white_background(original_image, binary_mask):
    # Invert the binary mask: 1 for lesion, 0 for background
    inverted_mask = 1 - binary_mask
    
    # Create a black lesion area
    black_lesion = np.zeros_like(original_image)
    
    # Combine the black lesion and the inverted mask to get a white background
    result_image = np.where(inverted_mask[..., None], 255, black_lesion)
    
    return result_image

# Define function to resize mask for display
def resize_mask_for_display(mask, display_size=(300, 300)):
    """
    Resize the segmentation mask for display on Streamlit.
    """
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Convert to an image
    return mask_image.resize(display_size)

# ---------------------Diagnosis classification---------------------

# Function to preprocess the image for subtype classification
def preprocess_for_subtype(image, size=256):
    # Assuming the image needs to be resized and normalized as in your subtype classification code
    image = image.resize((size, size))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Dictionary mapping the subtype abbreviations to full names
subtype_full_names = {
    'CYST': 'Cyst',
    'FA': 'Fibroadenoma',
    'LN': 'Lymph Node',
    'PAP': 'Papiloma',
    'DCIS': 'Ductal Carcinoma In Situ',
    'IDC': 'Invasive Ductal Carcinoma',
    'ILC': 'Invasive Lobular Carcinoma',
    'LP': 'Lymphoma',
    'UNK': 'Unknown'
}

def predict_subtype(model, image, is_malignant):
    processed_image = preprocess_for_subtype(image)
    prediction = model.predict(processed_image)
    predicted_subtype_index = np.argmax(prediction, axis=1)

    if is_malignant:
        subtype_mapping = ['DCIS', 'IDC', 'ILC', 'LP']
    else:
        subtype_mapping = ['CYST', 'FA', 'LN', 'PAP']
    
    predicted_subtype_abbreviation = subtype_mapping[predicted_subtype_index[0]]
    return subtype_full_names.get(predicted_subtype_abbreviation, 'Unknown')

# ---------------------Grad CAM---------------------

def get_img_array(img_path, size):
    # img_path is a file path in this case, but you'll need to modify it to work with an uploaded file if necessary
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(uploaded_file, heatmap, alpha=0.4, display_size=(300, 300)):
    # Load the original image
    original_img = Image.open(uploaded_file).convert('RGB')
    original_img = np.array(original_img)

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Invert the heatmap
    heatmap = 1 - heatmap  # Invert the heatmap colors

    # Convert the heatmap to a color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = heatmap * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Convert the superimposed image to PIL for resizing and display
    superimposed_img_pil = Image.fromarray(superimposed_img)
    
    # Resize the image for display
    resized_image = superimposed_img_pil.resize(display_size, Image.LANCZOS)

    return resized_image