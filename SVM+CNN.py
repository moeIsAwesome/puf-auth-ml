import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

# Build the CNN model


def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Get the input shape
input_shape = (img_height, img_width, 3)

# Build the model
cnn_model = build_cnn(input_shape)

# Call the model on a sample input to build it
cnn_model.build(input_shape=(None, img_height, img_width, 3))

# Train the model
history = cnn_model.fit(X_train, y_train, epochs=100, validation_data=(
    X_test, y_test), batch_size=batch_size)

# Save the trained CNN model
cnn_model.save('cnn_model.h5')

# Evaluate the model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
print(f'CNN Model Accuracy: {cnn_accuracy}')

# Extract features using the trained CNN model
feature_extractor = Model(inputs=cnn_model.input,
                          outputs=cnn_model.layers[-2].output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# Train an SVM classifier on the extracted features
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_features, y_train)

# Predict using the SVM classifier
y_pred = svm_classifier.predict(X_test_features)

# Evaluate the SVM classifier
svm_accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Classifier Accuracy: {svm_accuracy}')

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
