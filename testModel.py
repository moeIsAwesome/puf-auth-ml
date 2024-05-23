import joblib
import numpy as np

# Function to read .bin file and convert to flat array of bytes


def read_bin_file(file_path):
    with open(file_path, 'rb') as file:  # Open the file in binary mode
        byte_data = file.read()
        return np.frombuffer(byte_data, dtype=np.uint8)

# Pad or truncate sequence to a fixed length


def pad_or_truncate(sequence, length):
    if len(sequence) < length:
        return np.pad(sequence, (0, length - len(sequence)), 'constant')
    else:
        return sequence[:length]


# Fixed length for the input sequences
fixed_length = 1048585  # Define a suitable length based on your data

# Load the compressed model
loaded_model = joblib.load('svm_model_compressed.pkl')
print("Model loaded from disk")

# Function to make predictions on new PUF responses


def predict_puf_response(model, file_path):
    # Read the binary file and convert to a flat array of bytes
    response = read_bin_file(file_path)

    # Ensure the response is the correct length (pad or truncate if necessary)
    response = pad_or_truncate(response, fixed_length)

    # Reshape the response to match the expected input shape for the model
    response = response.reshape(1, -1)

    # Predict the PUF label
    prediction = model.predict(response)
    print(f"Prediction: {prediction}")

    # Map the prediction to the corresponding PUF label
    puf_labels = {0: 'PUF1', 1: 'PUF2', 2: 'PUF3'}
    predicted_label = puf_labels[prediction[0]]

    return predicted_label


# Example usage
new_puf_file = './datasets/dataset_ready/RPi2Dump/rpi2_0_aug3_2.bin'
predicted_puf = predict_puf_response(loaded_model, new_puf_file)
print(f"The response belongs to: {predicted_puf}")
