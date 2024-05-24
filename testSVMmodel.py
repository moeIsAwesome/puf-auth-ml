import joblib
import numpy as np

# Function to read .bin file and convert to flat array of bytes
def read_bin_file(file_path):
    with open(file_path, 'rb') as file:  # Open the file in binary mode
        byte_data = file.read()
        return np.frombuffer(byte_data, dtype=np.uint8)


# Load the compressed model
loaded_model = joblib.load('svm_model_compressed.pkl')


# Function to make predictions on new PUF responses
def predict_puf_response(model, file_path):
    # Read the binary file and convert to a flat array of bytes
    response = read_bin_file(file_path)

    # Reshape the response to match the expected input shape for the model
    response = response.reshape(1, -1)

    # Predict the PUF label
    prediction = model.predict(response)

    # Map the prediction to the corresponding PUF label
    puf_labels = {0: 'RPi1', 1: 'RPi2', 2: 'RPi3'}
    predicted_label = puf_labels[prediction[0]]
    print(predicted_label)
    return predicted_label


# Example usage
new_puf_file = './datasets/dataset_ready/RPi1Dump/rpi1_0_aug3_2.bin'
predicted_puf = predict_puf_response(loaded_model, new_puf_file)
