# TODO: Since the primary goal is to use the CNN solely for feature extraction and then train an SVM on those features, we don't need to save or evaluate the CNN model separately. The essential part is to extract the features from the CNN and then use those features for training the SVM. So, we can remove the code that saves and evaluates the CNN model.

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, Model
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Define paths to the folders
base_dir = './datasets/dataset_ready/datasets_images'
categories = ['RPi1Dump', 'RPi2Dump', 'RPi3Dump']

# Image parameters
img_height, img_width = 64, 64
batch_size = 32

# Load images and labels


def load_data(base_dir, categories, img_height, img_width):
    images = []
    labels = []
    for label, category in enumerate(categories):
        folder_path = os.path.join(base_dir, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)


# Load the data
images, labels = load_data(base_dir, categories, img_height, img_width)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Print the shapes of the datasets
print(f'Training data shape: {X_train.shape}')
print(f'Testing data shape: {X_test.shape}')
print(f'Training labels shape: {y_train.shape}')
print(f'Testing labels shape: {y_test.shape}')

# Build the CNN model using Functional API


def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Get the input shape
input_shape = (img_height, img_width, 3)

# Build the model
cnn_model = build_cnn(input_shape)

# Train the model
history = cnn_model.fit(X_train, y_train, epochs=10, validation_data=(
    X_test, y_test), batch_size=batch_size)

# Save the trained CNN model
cnn_model.save('cnn_model.keras')

# Evaluate the model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
print(f'CNN Model Accuracy: {cnn_accuracy}')

# Extract features from the CNN
feature_extractor = Model(inputs=cnn_model.input,
                          outputs=cnn_model.get_layer('flatten').output)

# Ensure the feature extractor model is built by calling it on some data
dummy_input = np.zeros((1, img_height, img_width, 3))
_ = feature_extractor.predict(dummy_input)

# Extract features
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# Train the SVM model on the extracted features
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_features, y_train)


# Save the trained model to a file with compression
model_filename = 'CNN+SVM_model_compressed.pkl'
# compress=3 is a reasonable trade-off between speed and size
joblib.dump(svm_model, model_filename, compress=3)

print(f"Compressed model saved to {model_filename}")
# Make predictions with the SVM
svm_predictions = svm_model.predict(X_test_features)

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f'SVM Model Accuracy: {svm_accuracy}')

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig('model_accuracy.png')  # Save the plot as an image file
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.savefig('model_loss.png')  # Save the plot as an image file
plt.show()
