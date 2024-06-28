import joblib
import numpy as np
import os

# Function to read .bin file and convert to flat array of bytes
def read_bin_file(file_path):
    with open(file_path, 'rb') as file:
        byte_data = file.read()
        return np.frombuffer(byte_data, dtype=np.uint8)

# Load the compressed model
model_filename = './reports/trained_with_augmented_length_64/model-trained-on-augmented-64.pkl'
loaded_model = joblib.load(model_filename)
print(f"Loaded model from {model_filename}")

# Function to make predictions on new PUF responses with confidence scores
def predict_puf_response(model, file_path):
    # Read the binary file and convert to a flat array of bytes
    response = read_bin_file(file_path)
    # Reshape the response to match the expected input shape for the model
    response = response.reshape(1, -1)
    # Predict the PUF label
    prediction = model.predict(response)
    prediction_proba = model.predict_proba(response)
    # Map the prediction to the corresponding PUF label
    puf_labels = {0: 'RPi1', 1: 'RPi2', 2: 'RPi3'}
    predicted_label = puf_labels[prediction[0]]
    confidence_score = prediction_proba[0][prediction[0]]
    return predicted_label, confidence_score

# Directory containing the files
directory = './dataset/random/rnd_64/'

# List to store confidence scores
confidences = []

# Iterate over the files in the directory
for filename in sorted(os.listdir(directory), key=lambda x: int(x[3:-4])):
    file_path = os.path.join(directory, filename)
    predicted_label, confidence = predict_puf_response(loaded_model, file_path)
    confidences.append(confidence)
    print(f"File: {filename}, Predicted Label: {predicted_label}, Confidence Score: {confidence:.2f}")

# Calculate average confidence
average_confidence = sum(confidences) / len(confidences)
print(f"Average Confidence: {average_confidence:.2f}")
