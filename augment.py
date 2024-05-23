import os
import shutil
import subprocess
import re

# Define the base directory where the datasets folder is located
base_dir = "./"
datasets_dir = os.path.join(base_dir, "datasets")
jar_file_path = os.path.join(datasets_dir, "pufmetrics-1.0-SNAPSHOT.jar")

# Define the augmentation parameters
augmentation_command = f"java -jar {jar_file_path} augment -a 5 -b 8388608 -f0 1024 -f1 1024 -c {{}}"

# List of PUF folders
puf_folders = ["RPi1Dump", "RPi2Dump", "RPi3Dump"]

# Number of bytes in the header
header_size = 9  # adjust if the header size is different


def add_header_to_augmented_file(file_path, header):
    with open(file_path, "rb") as aug_file:
        augmented_data = aug_file.read()

    # Prepend the header only if it's not already there
    if not augmented_data.startswith(header):
        with open(file_path, "wb") as aug_file:
            aug_file.write(header + augmented_data)


def augment_data():
    for puf_folder in puf_folders:
        folder_path = os.path.join(datasets_dir, puf_folder)
        augmented_folder_path = os.path.join(folder_path, "augmented")
        os.makedirs(augmented_folder_path, exist_ok=True)

        # Read the header from the original file once
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bin") and not re.search(r'_aug\d+\.bin$', file_name):
                src_file_path = os.path.join(folder_path, file_name)
                with open(src_file_path, "rb") as src_file:
                    header = src_file.read(header_size)
                break

        # Process each original file
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bin") and not re.search(r'_aug\d+\.bin$', file_name):
                src_file_path = os.path.join(folder_path, file_name)
                dest_file_path = os.path.join(augmented_folder_path, file_name)

                # Copy the original file to the destination
                shutil.copy(src_file_path, dest_file_path)

                # Augment the data
                command = augmentation_command.format(dest_file_path)
                print(f"Executing command: {command}")
                subprocess.run(command, shell=True, check=True)

        # Add header to all augmented files
        for file_name in os.listdir(augmented_folder_path):
            if re.match(r'.*_aug\d+\.bin$', file_name):
                augmented_file_path = os.path.join(
                    augmented_folder_path, file_name)
                add_header_to_augmented_file(augmented_file_path, header)

        # Delete the copied original files from the augmented folder
        for file_name in os.listdir(augmented_folder_path):
            if file_name.endswith(".bin") and not re.search(r'_aug\d+\.bin$', file_name):
                file_path_to_delete = os.path.join(
                    augmented_folder_path, file_name)
                os.remove(file_path_to_delete)


if __name__ == "__main__":
    augment_data()
