#Data Preprocessing:
import numpy as np
import os
import cv2
from PIL import Image

image_directory = 'C:/Users/wimuk/Desktop/Dataset_BUSI_with_GT/'

def load_images(image_folder, label_value):
    images = [img for img in os.listdir(image_directory + image_folder)]
    for image_name in images:
        if image_name.split('.')[1] == 'png' and '_mask' not in image_name:
            image = cv2.imread(image_directory + image_folder + image_name)
            if image is not None:
                image = Image.fromarray(image, 'RGB')
                image = image.resize((SIZE, SIZE))
                image = np.array(image)
                dataset.append(image)
                label.append(label_value)

SIZE = 224
dataset = []
label = []

load_images('benign/', 0)  # Benign class with label 0
load_images('malignant/', 1)  # Malignant class with label 1
load_images('normal/', 2)  # Normal class with label 2

# Convert dataset and label to numpy arrays
dataset = np.array(dataset)
label = np.array(label)
print("Dataset shape:", dataset.shape)
print("Label shape:", label.shape)


def custom_preprocessing(image):
    # Apply noise to the image
    noisy_image = add_noise_to_image(image)

    # Apply blur to the image
    blurred_image = apply_blur_to_image(noisy_image)

    # Adjust contrast and brightness
    enhanced_image = adjust_contrast_brightness(blurred_image)

    return enhanced_image


def add_noise_to_image(image):
    # Add noise to the image (customize this function as needed)
    noisy_image = np.clip(image + np.random.normal(loc=0, scale=0.1, size=image.shape), 0, 1)
    return noisy_image


def apply_blur_to_image(image):
    # Apply blur to the image (customize this function as needed)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image


def adjust_contrast_brightness(image):
    # Adjust contrast and brightness (customize as needed)
    img = (image * 255).astype(np.uint8)  # Convert image to uint8 for PIL
    pil_img = Image.fromarray(img)

    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced_img = enhancer.enhance(1.5)  # Increase contrast by a factor of 1.5

    enhancer = ImageEnhance.Brightness(enhanced_img)
    enhanced_img = enhancer.enhance(1.2)  # Increase brightness by a factor of 1.2

    enhanced_img = np.array(enhanced_img) / 255.0  # Convert back to float32

    return enhanced_img


# ResNet50:
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score

# Convert dataset and label to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split the dataset into train and test sets
from sklearn.model_selection import train_test_split

num_samples, height, width, channels = dataset.shape
X_flat = dataset.reshape(num_samples, -1)  # Reshape to (samples, height*width*channels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, label, test_size=0.25, random_state=42)

# Applying different augmentation settings to minority classes:
augmentation_class1 = ImageDataGenerator(
    # rescale=1./255,
    rotation_range=5,  # Rotate images by a maximum of 10 degrees
    width_shift_range=0.1,  # Shift images horizontally by 10% of the width
    height_shift_range=0.1,  # Shift images vertically by 10% of the height
    zoom_range=0.1,  # Zoom images by 10%
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=True,  # No vertical flipping
    preprocessing_function=custom_preprocessing
)
augmentation_class2 = ImageDataGenerator(
    rotation_range=30,  # Rotate images by a maximum of 10 degrees
    width_shift_range=0.2,  # Shift images horizontally by 10% of the width
    height_shift_range=0.2,  # Shift images vertically by 10% of the height
    zoom_range=0.2,  # Zoom images by 10%
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=True,
    preprocessing_function=custom_preprocessing

)

X_train = X_train.reshape(-1, 224, 224, 3)  # Reshape your input data to match the expected input shape

datagen = ImageDataGenerator(
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=True,  # Flip images vertically
    fill_mode='nearest'  # Fill in missing pixels using the nearest available
)
datagen.fit(X_train)
augmented_images = []
augmented_labels = []

# Number of times to augment the data (in this case, we'll double the dataset)
augmentation_factor = 2

for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False):
    augmented_images.append(x_batch)
    augmented_labels.append(y_batch)
    if len(augmented_images) >= augmentation_factor:
        break

# Concatenate the augmented data batches
X_train = np.concatenate(augmented_images)
y_train = np.concatenate(augmented_labels)

# Verify the shape of augmented data
print("Shape of augmented images:", X_train.shape)
print("Shape of augmented labels:", y_train.shape)


def apply_augmentation(X_train, y_train):
    if y_train == 1:  # Check for class 1
        return augmentation_class1.random_transform(X_train), y_train
    if y_train == 2:
        return augmentation_class2.random_transform(X_train), y_train
    else:
        return X_train, y_train


X_test = X_test.reshape(-1, 224, 224, 3)  # Reshape your input data to match the expected input shape

from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_labels = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
# Assign more weight to the second class
class_weights[1] *= 10.0  # You can adjust this factor as needed to assign more weight
class_weights[2] *= 4.0  # You can adjust this factor as needed to assign more weight
# Create a dictionary of class weights
class_weight = {i: weight for i, weight in enumerate(class_weights)}

INPUT_SHAPE = (224, 224, 3)
inp = keras.layers.Input(shape=INPUT_SHAPE)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

pretrained_model = keras.applications.ResNet50(include_top=False,
                                               input_shape=(224, 224, 3),
                                               pooling='max', classes=3,
                                               weights='imagenet')

model.add(pretrained_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())  # Batch Normalization layer
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# pretrained_model.trainable=False
for layer in pretrained_model.layers:
    layer.trainable = False
# Unfreeze the last few layers for fine-tuning(I achieved much better results adding this step)
for layer in pretrained_model.layers[-12:]:
    layer.trainable = True

###Learning rate:
from tensorflow.keras.callbacks import LearningRateScheduler

# Learning rate schedule
# def lr_schedule(epoch):
#    initial_learning_rate = 0.001
#    decay = 0.9
#    lr = initial_learning_rate * decay ** epoch
#    return lr

# lr_scheduler = LearningRateScheduler(lr_schedule)
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# Adding callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = 'weights/best_model_weights.h5'
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max',
                             save_weights_only=True)
# Fit the model
history = model.fit(np.array(X_train),
                    y_train,
                    batch_size=32,
                    verbose=1,
                    epochs=50,
                    # validation_split = 0.1,
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    class_weight=class_weight,
                    callbacks=[checkpoint]
                    # callbacks=[reduce_lr]
                    )
# Save the entire model
model.save('breast_cancer_model.h5')
model.load_weights(checkpoint_path)

#Model Evaluation:
# Import necessary libraries
from sklearn.metrics import classification_report, confusion_matrix

# Assuming the model has been trained using the provided code, and `model` refers to the trained model

# Evaluate the model on the test set
evaluation = model.evaluate(X_test, y_test, verbose=1)

# Generate predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Classification report and confusion matrix
print("Test Accuracy:", evaluation[1])
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))
print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

import numpy as np
import matplotlib.pyplot as plt

# Get some random indices from the test set
num_samples_to_display = 5  # Number of random samples to display
random_indices = np.random.choice(X_test.shape[0], num_samples_to_display, replace=False)

# Get the corresponding images, ground truth labels, and predicted labels
images_to_display = X_test[random_indices]
true_labels = y_test[random_indices]
predicted_labels = model.predict(images_to_display)
predicted_labels = np.argmax(predicted_labels, axis=1)

# Convert true_labels to integers because they are one-hot encoded
true_labels_int = np.argmax(true_labels, axis=1)

# Display the images along with their ground truth and predicted labels
plt.figure(figsize=(15, 5))
for i in range(num_samples_to_display):
    plt.subplot(1, num_samples_to_display, i + 1)
    plt.imshow(images_to_display[i].reshape(height, width, channels))
    plt.title(f"True: {true_labels_int[i]}, Predicted: {predicted_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

