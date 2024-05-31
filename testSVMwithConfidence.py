
# predict_model_with_analysis.py

import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to read .bin file and convert to flat array of bytes
def read_bin_file(file_path):
    with open(file_path, 'rb') as file:  # Open the file in binary mode
        byte_data = file.read()
        return np.frombuffer(byte_data, dtype=np.uint8)


# Load the compressed model
model_filename = 'svm_model_compressed.pkl'
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

# Function to evaluate model on corrupted data


def evaluate_corrupted_data(corr_base_path):
    results = []
    true_labels = {
        'rpi1_0': 'RPi1', 'rpi2_0': 'RPi2', 'rpi3_0': 'RPi3',
        'rpi1_0_corrBot': 'RPi1', 'rpi1_0_corrMid': 'RPi1', 'rpi1_0_corrTop': 'RPi1',
        'rpi2_0_corrBot': 'RPi2', 'rpi2_0_corrMid': 'RPi2', 'rpi2_0_corrTop': 'RPi2',
        'rpi3_0_corrBot': 'RPi3', 'rpi3_0_corrMid': 'RPi3', 'rpi3_0_corrTop': 'RPi3'
    }

    for level in range(10, 100, 10):
        folder_name = f"{level}-percent-corrupted"
        folder_path = os.path.join(corr_base_path, folder_name)
        accuracy_count = 0
        total_files = 0
        confidence_scores = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bin"):
                file_path = os.path.join(folder_path, file_name)
                predicted_label, confidence = predict_puf_response(
                    loaded_model, file_path)
                true_label = true_labels[file_name.replace('.bin', '')]
                if predicted_label == true_label:
                    accuracy_count += 1
                total_files += 1
                confidence_scores.append(confidence)

        accuracy = accuracy_count / total_files if total_files > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        results.append((level, accuracy, avg_confidence))

        print(
            f"{level}% corruption - Accuracy: {accuracy:.2f}, Average Confidence: {avg_confidence:.2f}")

    return results

# Plotting the results


def plot_results(results):
    corruption_levels = [x[0] for x in results]
    accuracies = [x[1] for x in results]
    confidences = [x[2] for x in results]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(corruption_levels, accuracies, marker='o')
    plt.title('Accuracy vs Corruption Level')
    plt.xlabel('Corruption Level (%)')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(corruption_levels, confidences, marker='o', color='orange')
    plt.title('Confidence vs Corruption Level')
    plt.xlabel('Corruption Level (%)')
    plt.ylabel('Average Confidence')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    corr_base_path = './dataset/data/CorrTest'
    results = evaluate_corrupted_data(corr_base_path)
    plot_results(results)
