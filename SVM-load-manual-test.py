import joblib
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read .bin file and convert to flat array of bytes
def read_bin_file(file_path):
    with open(file_path, 'rb') as file:  # Open the file in binary mode
        byte_data = file.read()
        return np.frombuffer(byte_data, dtype=np.uint8)

# Load data
def load_data(folder_path, label):
    data = []
    labels = []
    for file_name in tqdm(os.listdir(folder_path), desc=f"Loading {folder_path}"):
        file_path = os.path.join(folder_path, file_name)
        file_data = read_bin_file(file_path)
        # Debugging statement
        print(f"Loaded {file_path} with shape {file_data.shape}")
        data.append(file_data)
        labels.append(label)
    return data, labels

# Paths to folders for test data
test_folder_paths = ["./dataset/data/test/RPi1Dump/", "./dataset/data/test/RPi2Dump/", "./dataset/data/test/RPi3Dump/", "./dataset/data/test/Unknown/"]
test_labels = [0, 1, 2, 3]  # Updated with new class label

# Read test data from all folders
test_data = []
test_labels_all = []
for folder_path, label in zip(test_folder_paths, test_labels):
    data, label = load_data(folder_path, label)
    test_data.extend(data)
    test_labels_all.extend(label)

# Convert test data to numpy arrays
try:
    test_data = np.array(test_data)
    test_labels_all = np.array(test_labels_all)
except ValueError as e:
    print("Error converting to numpy arrays:", e)
    for i, data in enumerate(test_data):
        print(f"Data at index {i} has shape {data.shape}")

# Load the pre-trained model
model_filename = 'model-trained-on-augmented-with-4-classes-full-length.pkl'
svm_model = joblib.load(model_filename)
print(f"Loaded model from {model_filename}")

# Test the model
y_test_pred = svm_model.predict(test_data)
test_accuracy = accuracy_score(test_labels_all, y_test_pred)
test_precision = precision_score(test_labels_all, y_test_pred, average='weighted')
test_recall = recall_score(test_labels_all, y_test_pred, average='weighted')
test_f1 = f1_score(test_labels_all, y_test_pred, average='weighted')

# Calculate confidence scores
test_probabilities = svm_model.predict_proba(test_data)
test_confidences = np.max(test_probabilities, axis=1)
average_confidence = np.mean(test_confidences)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Precision: {test_precision * 100:.2f}%")
print(f"Test Recall: {test_recall * 100:.2f}%")
print(f"Test F1 Score: {test_f1 * 100:.2f}%")
print(f"Average Confidence: {average_confidence * 100:.2f}%")

# Print each test instance's confidence
for i, confidence in enumerate(test_confidences):
    print(f"Instance {i}: Confidence = {confidence * 100:.2f}%")

# Visualizations
def plot_metrics(y_true, y_pred, dataset_type, confidences):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    avg_confidence = np.mean(confidences)

    metrics = [accuracy, precision, recall, f1, avg_confidence]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Avg Confidence']

    plt.figure(figsize=(10, 5))
    plt.bar(metric_names, metrics, color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.ylim(0, 1)
    plt.title(f'{dataset_type} Set Metrics')
    plt.ylabel('Score')
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v * 100:.2f}%", ha='center', fontweight='bold')
    plt.show()

# Plot test metrics
plot_metrics(test_labels_all, y_test_pred, "Test", test_confidences)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, dataset_type):
    label_mapping = {0: 'RPi1', 1: 'RPi2', 2: 'RPi3', 3: 'Unknown'}  # Updated with new class label
    y_true_mapped = [label_mapping[label] for label in y_true]
    y_pred_mapped = [label_mapping[label] for label in y_pred]
    
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=['RPi1', 'RPi2', 'RPi3', 'Unknown'])  # Updated with new class label
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['RPi1', 'RPi2', 'RPi3', 'Unknown'], yticklabels=['RPi1', 'RPi2', 'RPi3', 'Unknown'])  # Updated with new class label
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{dataset_type} Set Confusion Matrix')
    plt.show()

# Plot test confusion matrix
plot_confusion_matrix(test_labels_all, y_test_pred, "Test")
