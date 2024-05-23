import os
import shutil
import subprocess
import re  # Import the re module for regular expressions

# Define the base directory where the datasets folder is located
base_dir = "./"
datasets_dir = os.path.join(base_dir, "datasets")
jar_file_path = os.path.join(datasets_dir, "pufmetrics-1.0-SNAPSHOT.jar")

# Define the corruption parameters
corruption_command = f"java -jar {jar_file_path} corrupt -b 8388608 -p 1 -c {{}}"

# List of PUF folders
puf_folders = ["RPi1Dump", "RPi2Dump", "RPi3Dump"]

# Number of bytes in the header
header_size = 9  # adjust if the header size is different


def add_header_to_corrupted_files(folder_path, header):
    corrupted_folder_path = os.path.join(folder_path, "corrupted")
    for file_name in os.listdir(corrupted_folder_path):
        if file_name.endswith(".bin"):
            corrupted_file_path = os.path.join(
                corrupted_folder_path, file_name)
            with open(corrupted_file_path, "rb") as corrupt_file:
                corrupted_data = corrupt_file.read()
            # Prepend the header only if it's not already there
            if not corrupted_data.startswith(header):
                with open(corrupted_file_path, "wb") as corrupt_file:
                    corrupt_file.write(header + corrupted_data)


def corrupt_data():
    for puf_folder in puf_folders:
        folder_path = os.path.join(datasets_dir, puf_folder)
        corrupted_folder_path = os.path.join(folder_path, "corrupted")
        os.makedirs(corrupted_folder_path, exist_ok=True)

        # Read the header from the original file once
        header = None
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bin") and not re.search(r'_corr(Bot|Mid|Top)\.bin$', file_name):
                src_file_path = os.path.join(folder_path, file_name)
                with open(src_file_path, "rb") as src_file:
                    header = src_file.read(header_size)
                break

        # Process each original file
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bin") and not re.search(r'_corr(Bot|Mid|Top)\.bin$', file_name):
                src_file_path = os.path.join(folder_path, file_name)
                dest_file_path = os.path.join(corrupted_folder_path, file_name)

                # Copy the original file to the destination
                shutil.copy(src_file_path, dest_file_path)

                # Corrupt the data
                command = corruption_command.format(dest_file_path)
                print(f"Executing command: {command}")
                subprocess.run(command, shell=True, check=True)

        # Add header to all corrupted files
        add_header_to_corrupted_files(folder_path, header)

        # Delete the copied original files from the corrupted folder
        for file_name in os.listdir(corrupted_folder_path):
            if file_name.endswith(".bin") and not re.search(r'_corr(Bot|Mid|Top)\.bin$', file_name):
                file_path_to_delete = os.path.join(
                    corrupted_folder_path, file_name)
                os.remove(file_path_to_delete)


if __name__ == "__main__":
    corrupt_data()
