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


# Paths to folders for training
train_folder_paths = ["./dataset/data/RPi1Dump", "./dataset/data/RPi2Dump"]
train_labels = [0, 1]

# Read data from training folders
train_data = []
train_labels_list = []
for folder_path, label in zip(train_folder_paths, train_labels):
    data, label = load_data(folder_path, label)
    train_data.extend(data)
    train_labels_list.extend(label)

# Convert to numpy arrays for training
try:
    train_data = np.array(train_data)
    train_labels_list = np.array(train_labels_list)
except ValueError as e:
    print("Error converting to numpy arrays:", e)
    for i, data in enumerate(train_data):
        print(f"Data at index {i} has shape {data.shape}")

# Split the training data into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_labels_list, test_size=0.2, random_state=42)

# Create and train the SVM model with probability=True
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train, y_train)

# Validate the model
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(
    y_val, y_val_pred, average='weighted', zero_division=1)
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Precision: {val_precision * 100:.2f}%")
print(f"Validation Recall: {val_recall * 100:.2f}%")
print(f"Validation F1 Score: {val_f1 * 100:.2f}%")

# Paths to folders for testing
test_folder_path = "./dataset/data/RPi3Dump"

# Read data from test folder (RPi3Dump)
# No labels provided for test data
test_data, _ = load_data(test_folder_path, None)

# Convert to numpy arrays for testing
try:
    test_data = np.array(test_data)
except ValueError as e:
    print("Error converting to numpy arrays:", e)
    for i, data in enumerate(test_data):
        print(f"Test data at index {i} has shape {data.shape}")

# Predict and analyze confidence scores on the test data
test_probabilities = svm_model.predict_proba(test_data)
test_confidences = np.max(test_probabilities, axis=1)
average_confidence = np.mean(test_confidences)

# Predict class labels for the test data
test_predictions = svm_model.predict(test_data)

# Mapping class indices to label names
label_mapping = {0: 'RPi1', 1: 'RPi2'}

print(f"Average Test Confidence: {average_confidence * 100:.2f}%")

# Print each test instance's confidence and predicted class with label names
for i, (confidence, prediction) in enumerate(zip(test_confidences, test_predictions)):
    print(
        f"Test Instance {i}: Confidence = {confidence * 100:.2f}%, Predicted Class = {label_mapping[prediction]}")

# Function to plot the confidence scores


def plot_confidence_distribution(confidences, title):
    plt.figure(figsize=(10, 5))
    plt.hist(confidences, bins=20, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.show()


# Plot confidence distribution for the test data
plot_confidence_distribution(
    test_confidences, "Confidence Distribution for Unlabeled Test Data (RPi3Dump)")
