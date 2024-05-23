import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to read .bin file and convert to flat array of bytes


def read_bin_file(file_path):
    with open(file_path, 'rb') as file:  # Open the file in binary mode
        byte_data = file.read()
        return np.frombuffer(byte_data, dtype=np.uint8)

# Load data


def load_data(folder_path, label):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        data.append(read_bin_file(file_path))
        labels.append(label)
    return data, labels


# Paths to folders
folder_paths = ["./datasets/dataset_ready/RPi1Dump",
                "./datasets/dataset_ready/RPi2Dump", "./datasets/dataset_ready/RPi3Dump"]
labels = [0, 1, 2]

# Read data from all folders
all_data = []
all_labels = []
for folder_path, label in zip(folder_paths, labels):
    data, label = load_data(folder_path, label)
    all_data.extend(data)
    all_labels.extend(label)

# Convert to numpy arrays
all_data = np.array(all_data)
all_labels = np.array(all_labels)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    all_data, all_labels, test_size=0.2, random_state=42)

# Further split the training set to create a validation set (20% of training set)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Validate the model
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Test the model
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Save the trained model to a file
model_filename = 'svm_model.pkl'
joblib.dump(svm_model, model_filename)
print(f"Model saved to {model_filename}")
