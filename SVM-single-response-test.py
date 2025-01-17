import joblib
import numpy as np

# Function to read .bin file and convert to flat array of bytes
def read_bin_file(file_path):
    with open(file_path, 'rb') as file:  # Open the file in binary mode
        byte_data = file.read()
        return np.frombuffer(byte_data, dtype=np.uint8)

# Load the compressed model
model_filename = './reports/trained_with_augmented_4_classes_full_length/model-trained-on-augmented-with-4-classes-full-length.pkl'
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
    print(f"Prediction Probabilities: {prediction_proba}")
    # Map the prediction to the corresponding PUF label
    puf_labels = {0: 'RPi1', 1: 'RPi2', 2: 'RPi3', 3: 'Unknown'}  # Updated with new class label
    predicted_label = puf_labels[prediction[0]]
    confidence_score = prediction_proba[0][prediction[0]]
    print(f"Predicted Label: {predicted_label}, Confidence Score: {confidence_score:.2f}")
    return predicted_label, confidence_score

# Example usage
new_puf_file = './dataset/data/Unknown/all0.bin'
predicted_puf, confidence = predict_puf_response(loaded_model, new_puf_file)
