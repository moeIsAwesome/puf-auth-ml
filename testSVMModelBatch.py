import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the trained model
model_filename = 'svm_model_compressed.pkl'
svm_model = joblib.load(model_filename)

# Function to read .bin file and convert to flat array of bytes
def read_bin_file(file_path):
    with open(file_path, 'rb') as file:  # Open the file in binary mode
        byte_data = file.read()
        return np.frombuffer(byte_data, dtype=np.uint8)

# Function to evaluate the model on new data
def evaluate_model(new_data_folder_paths, true_labels):
    new_data = []
    for folder_path in new_data_folder_paths:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            file_data = read_bin_file(file_path)
            new_data.append(file_data)

    # Convert to numpy array
    new_data = np.array(new_data)
    true_labels = np.array(true_labels)

    # Predict using the trained model
    predictions = svm_model.predict(new_data)
    prediction_probabilities = svm_model.predict_proba(new_data)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Calculate and print confidence metrics
    confidences = np.max(prediction_probabilities, axis=1)
    average_confidence = np.mean(confidences)
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        print(f"Prediction: {'RP' + str(pred+1)}, Confidence: {conf:.2f}")

    print(f"Average Confidence: {average_confidence:.2f}")

    # Map numerical labels to string labels
    label_mapping = {0: 'RP1', 1: 'RP2', 2: 'RP3'}
    true_labels_str = [label_mapping[label] for label in true_labels]
    predictions_str = [label_mapping[pred] for pred in predictions]

    # Plot confusion matrix
    cm = confusion_matrix(true_labels_str, predictions_str, labels=['RP1', 'RP2', 'RP3'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['RP1', 'RP2', 'RP3'], yticklabels=['RP1', 'RP2', 'RP3'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for New Data')
    plt.show()

# Example usage
new_data_folder_paths = ["./dataset/data/RPi1Dump/", "./dataset/data/RPi2Dump/", "./dataset/data/RPi3Dump/"]
true_labels = [0] * len(os.listdir("./dataset/data/RPi1Dump/")) + [1] * len(os.listdir("./dataset/data/RPi2Dump/")) + [2] * len(os.listdir("./dataset/data/RPi3Dump/"))

evaluate_model(new_data_folder_paths, true_labels)
