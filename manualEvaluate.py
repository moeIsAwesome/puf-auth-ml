# To use this script create three folders named RPI1, RPI2, and RPI3 in the dataset/data folder. Each folder should contain the corresponding PUF responses for the Raspberry Pi device.

# evaluate_model_with_confidence.py

import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

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
    return predicted_label, confidence_score, prediction_proba[0]

# Function to evaluate the model on a dataset


def evaluate_model_on_dataset(base_path):
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    all_confidences = []

    puf_labels = {'RPi1': 0, 'RPi2': 1, 'RPi3': 2}
    inverse_puf_labels = {0: 'RPi1', 1: 'RPi2', 2: 'RPi3'}

    for folder_name in puf_labels.keys():
        folder_path = os.path.join(base_path, folder_name)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bin"):
                file_path = os.path.join(folder_path, file_name)
                true_labels.append(folder_name)
                predicted_label, confidence, confidences = predict_puf_response(
                    loaded_model, file_path)
                predicted_labels.append(predicted_label)
                confidence_scores.append(confidence)
                all_confidences.append(confidences)

    # Convert true labels to numerical form for metrics calculation
    true_labels_num = [puf_labels[label] for label in true_labels]
    predicted_labels_num = [puf_labels[label] for label in predicted_labels]

    # Calculate metrics
    accuracy = accuracy_score(true_labels_num, predicted_labels_num)
    precision = precision_score(
        true_labels_num, predicted_labels_num, average='weighted')
    recall = recall_score(
        true_labels_num, predicted_labels_num, average='weighted')
    f1 = f1_score(true_labels_num, predicted_labels_num, average='weighted')
    avg_confidence = np.mean(confidence_scores)
    cm = confusion_matrix(true_labels_num, predicted_labels_num)

    # Calculate Brier score for each class and average them
    brier_scores = []
    for class_index in range(len(puf_labels)):
        class_true = [1 if label ==
                      class_index else 0 for label in true_labels_num]
        class_prob = [prob[class_index] for prob in all_confidences]
        brier_scores.append(brier_score_loss(class_true, class_prob))
    avg_brier = np.mean(brier_scores)

    # Calibration curve for each class
    plt.figure(figsize=(8, 6))
    for class_index in range(len(puf_labels)):
        class_true = [1 if label ==
                      class_index else 0 for label in true_labels_num]
        class_prob = [prob[class_index] for prob in all_confidences]
        prob_true, prob_pred = calibration_curve(
            class_true, class_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o',
                 label=f'Class {inverse_puf_labels[class_index]}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()

    # Print metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Brier Score: {avg_brier:.2f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=inverse_puf_labels.values(
    ), yticklabels=inverse_puf_labels.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot metrics
    metrics = [accuracy, precision, recall, f1, avg_confidence]
    metric_names = ['Accuracy', 'Precision',
                    'Recall', 'F1 Score', 'Avg Confidence']

    plt.figure(figsize=(12, 6))
    plt.bar(metric_names, metrics, color=[
            'blue', 'orange', 'green', 'red', 'purple'])
    plt.ylim(0, 1)
    plt.title('Model Metrics')
    plt.ylabel('Score')
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.show()


if __name__ == "__main__":
    base_path = './dataset/data'
    evaluate_model_on_dataset(base_path)
