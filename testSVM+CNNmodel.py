import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import joblib

# Load the trained models
cnn_model = load_model('cnn_model.keras')
svm_model = joblib.load('CNN+SVM_model_compressed.pkl')

# Function to preprocess the input image


def preprocess_image(image_path, img_height=64, img_width=64):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Extract features from the CNN


def extract_features(model, img_array):
    feature_extractor = Model(
        inputs=model.input, outputs=model.get_layer('flatten').output)
    features = feature_extractor.predict(img_array)
    return features

# Function to make a prediction


def predict(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Extract features
    features = extract_features(cnn_model, img_array)

    # Make a prediction with the SVM
    prediction = svm_model.predict(features)

    # Map the prediction to the category
    categories = ['RPi1Dump', 'RPi2Dump', 'RPi3Dump']
    predicted_category = categories[prediction[0]]

    return predicted_category


if __name__ == "__main__":
    # Specify the image path here
    # Update this path to your image file
    image_path = './datasets/dataset_ready/datasets_images/RPi1Dump/rpi1_18_corrMid.png'
    category = predict(image_path)
    print(f'The image belongs to category: {category}')
