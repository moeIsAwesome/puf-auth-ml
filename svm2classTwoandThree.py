import joblib
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Paths to folders
folder_paths = ["./dataset/data/RPi2Dump", "./dataset/data/RPi3Dump"]
labels = [0, 1]

# Read data from all folders
all_data = []
all_labels = []
for folder_path, label in zip(folder_paths, labels):
    data, label = load_data(folder_path, label)
    all_data.extend(data)
    all_labels.extend(label)

# Convert to numpy arrays
try:
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
except ValueError as e:
    print("Error converting to numpy arrays:", e)
    for i, data in enumerate(all_data):
        print(f"Data at index {i} has shape {data.shape}")

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

# Further split the training set to create a validation set (20% of training set)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create and train the SVM model with probability=True
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train, y_train)

# Validate the model
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
# print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
# print(f"Validation Precision: {val_precision * 100:.2f}%")
# print(f"Validation Recall: {val_recall * 100:.2f}%")
# print(f"Validation F1 Score: {val_f1 * 100:.2f}%")

# Test the model
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Calculate confidence scores
test_probabilities = svm_model.predict_proba(X_test)
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

# Save the trained model to a file with compression
model_filename = 'svm_model_compressed.pkl'
# compress=3 is a reasonable trade-off between speed and size
joblib.dump(svm_model, model_filename, compress=3)
print(f"Compressed model saved to {model_filename}")

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

# Plot validation metrics
# plot_metrics(y_val, y_val_pred, "Validation", np.max(svm_model.predict_proba(X_val), axis=1))

# Plot test metrics
plot_metrics(y_test, y_test_pred, "Test", test_confidences)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, dataset_type):
    label_mapping = {0: 'RPi2', 1: 'RPi3'}
    y_true_mapped = [label_mapping[label] for label in y_true]
    y_pred_mapped = [label_mapping[label] for label in y_pred]
    
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=['RPi2', 'RPi3'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['RPi2', 'RPi3'], yticklabels=['RPi2', 'RPi3'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{dataset_type} Set Confusion Matrix')
    plt.show()

# Plot validation confusion matrix
# plot_confusion_matrix(y_val, y_val_pred, "Validation")

# Plot test confusion matrix
plot_confusion_matrix(y_test, y_test_pred, "Test")
