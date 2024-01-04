"""# **Model Comparison**"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from imgaug import augmenters as iaa

import tensorflow as tf
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from keras.applications import DenseNet201, ResNet101
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Concatenate
from keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras.metrics import AUC, F1Score, Precision, Recall

base_path = 'C:/Users/wimuk/Desktop/Updated_combine_datasets'

def count_dataset_instances(file_paths, base_path):
    count_dataset = sum([1 for path in file_paths if base_path in path])
    return count_dataset

# Load data from the first dataset, excluding mask images
X, y = [], []
categories = os.listdir(base_path)
for category in categories:
    class_num = categories.index(category)
    path = os.path.join(base_path, category)
    for img in os.listdir(path):
        # if '_mask' not in img:  # Skip mask images
            X.append(os.path.join(base_path, category, img))  # Include the base path
            y.append(class_num)

# Convert labels to categorical
y = to_categorical(y)

# Count instances from each dataset
count_dataset = count_dataset_instances(X, base_path)
print(f"Total instances: Dataset - {count_dataset}")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, # feature dataset (independent variables or inputs)
                                                  y, # target dataset (dependent variables or outputs)
                                                  test_size=0.2, # 20% validation, remaining 80% training
                                                  random_state=42,
                                                  stratify=y)

# Count instances in training set
train_count_dataset = count_dataset_instances(X_train, base_path)
print(f"Training set: Dataset - {train_count_dataset} instances")

# Count instances in validation set
val_count_dataset = count_dataset_instances(X_val, base_path)
print(f"Validation set: Dataset 1 - {val_count_dataset} instances")

def calculate_augmentation_multiplier(current_count, target_count):
    if current_count >= target_count:
        return 1
    else:
        return int(np.ceil(target_count / current_count))

# Set a common target count for all classes
target_count = 1000  # Example: set to a value that suits your dataset

# Calculate multipliers
benign_multiplier = calculate_augmentation_multiplier(648, target_count)
malignant_multiplier = calculate_augmentation_multiplier(414, target_count)
normal_multiplier = calculate_augmentation_multiplier(133, target_count)

class_augmentation_multipliers = {
    'benign': benign_multiplier,
    'malignant': malignant_multiplier,
    'normal': normal_multiplier
}

augmentation_counts = {
    'benign': 0,
    'malignant': 0,
    'normal': 0
}

# custom generator is designed for batch processing of image data
class CustomDataGenerator(Sequence):
    # Initialize with a single directory since paths are absolute
    def __init__(self, image_filenames, labels, batch_size, img_size, datagen=None, color_mode='grayscale', imgaug_augmentations=None):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.datagen = datagen
        self.imgaug_augmentations = imgaug_augmentations
        self.color_mode = color_mode
        # self.use_augmentation = use_augmentation

    def __len__(self): # calculates the total number of batches the generator will produce
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def _visualize_batch(self, images, titles): # how data is being processed and augmented
        """Visualizes a batch of images."""
        plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(4, 4, i + 1)  # Adjust grid size depending on batch size
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        labels = []
        for file_name, label in zip(batch_x, batch_y):
            img = cv2.imread(file_name)  # file_name includes the full path

            if img is not None:
                if self.color_mode == 'grayscale':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img / 255.0

                class_label = np.argmax(label)  # Assuming one-hot encoded labels
                class_name = categories[class_label]  # Map the label index to class name

                # if self.use_augmentation:
                multiplier = class_augmentation_multipliers[class_name]
                # Apply augmentations based on the multiplier
                for _ in range(multiplier):
                    if self.datagen:
                        img_augmented = self.datagen.random_transform(img)
                    elif self.imgaug_augmentations:
                        img_augmented = self.imgaug_augmentations.augment_image(img)
                    else:
                        img_augmented = img  # No augmentation applied

                    images.append(img_augmented)
                    labels.append(label)
                    augmentation_counts[class_name] += 1  # Increment the count

        # print(f"Number of images in the batch: {len(images)}")
        return np.array(images), np.array(labels)

img_size = 256
batch_size = 32

# Generate a batch of images
batch_images, _ = next(iter(CustomDataGenerator(X, y, batch_size, img_size)))

# Plotting
plt.figure(figsize=(8, 8))
for i in range(9):  # Display the first 9 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(batch_images[i])
    plt.axis('off')
plt.show()

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,  # Keeping rotation realistic
    width_shift_range=0.02,  # Slightly increased shift range for more variability
    height_shift_range=0.02,  # Slightly increased shift range
    shear_range=0.1,  # Retain minor distortions
    zoom_range=0.1,  # Maintain important features with minor zoom
    horizontal_flip=True,  # Adding horizontal flip
    vertical_flip=True,  # Adding vertical flip
    fill_mode='nearest'  # Fill mode remains the same
)

advanced_augmentations = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=30, sigma=5)),  # Elastic deformation with reduced frequency
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=0.0005*255)),  # Reduced noise intensity and frequency
    iaa.CLAHE(clip_limit=2),  # CLAHE remains the same for contrast adjustment
    iaa.LinearContrast((0.95, 1.05)),  # More subtle linear contrast adjustment
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5)))  # Optional: Adding occasional Gaussian blur for variety
])

# Create data generators
train_generator = CustomDataGenerator(X_train, y_train, batch_size, img_size, datagen=datagen, imgaug_augmentations=advanced_augmentations)
val_generator = CustomDataGenerator(X_val, y_val, batch_size, img_size)  # No imgaug_augmentations for validation data

# Iterate over the entire training set to count augmentations
for _ in range(len(train_generator)):
    batch_images, batch_labels = train_generator.__getitem__(_)

# Print the augmentation counts
print("Augmented Counts:")
for class_name, count in augmentation_counts.items():
    print(f"{class_name}: {count} images")

"""**ResNet101**"""

base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
predictions = Dense(len(categories), activation='softmax')(x)
model_ResNet101 = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=0.0001)
# optimizer = SGD(learning_rate=0.001, momentum=0.9)
model_ResNet101.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# model_ResNet101.compile(optimizer=optimizer, loss='categorical_crossentropy',
#               metrics=['accuracy', Precision(), Recall(), AUC()])

# Model summary
# model_DenseNet201.summary()

# ModelCheckpoint callback
# checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.1
    else:
        return lr * 0.01

lr_scheduler = LearningRateScheduler(scheduler)

# Add all the desired callbacks
callbacks_list = [early_stopping, reduce_lr, lr_scheduler]

# Fit the model_DenseNet201
history_ResNet101 = model_ResNet101.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=val_generator,
    validation_steps=len(X_val) // batch_size,
    callbacks=callbacks_list  # Add the callbacks here
)

# Load the best model
# model_ResNet101.load_model('best_model.h5')

# Evaluate the model on the validation data
val_loss, val_accuracy, val_precision, val_recall, val_auc = model_ResNet101.evaluate(val_generator, steps=len(X_val) // batch_size)

# Print evaluation metrics
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
# print(f"Validation Precision: {val_precision}")
# print(f"Validation Recall: {val_recall}")
# print(f"Validation AUC: {val_auc}")