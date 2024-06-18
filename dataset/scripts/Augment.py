import os
import shutil
import subprocess
import re
import time

# Define the base directory where the datasets folder is located
datasets_dir = "./dataset/data"
jar_file_path = "./dataset/scripts/puf-tool.jar"
f_0 = 8192
f_1 = 64

# Define the augmentation parameters
augmentation_command = f"java -jar {jar_file_path} augment -a 5 -b 8388608 -f0 {f_0} -f1 {f_1} {{}}"

# List of PUF folders
puf_folders = ["RPi1Dump", "RPi2Dump", "RPi3Dump"]

def augment_data():
    for puf_folder in puf_folders:
        folder_path = os.path.join(datasets_dir, puf_folder)
        augmented_folder_path = os.path.join(folder_path, f"{f_0}-{f_1}-augmented")
        os.makedirs(augmented_folder_path, exist_ok=True)

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

                # Assuming the augmentation tool generates files with a pattern like _aug1.bin, _aug2.bin, etc.
                # Adjust range if the number of augmentations is different
                for aug_index in range(1, 6):  # Assuming it generates aug1 to aug5
                    augmented_file_name = file_name.replace(".bin", f"_aug{aug_index}.bin")
                    augmented_file_path = os.path.join(augmented_folder_path, augmented_file_name)
                    if not os.path.exists(augmented_file_path):
                        print(f"Augmented file {augmented_file_path} does not exist, retrying...")
                        time.sleep(1)  # Wait a bit and try again
                        if not os.path.exists(augmented_file_path):
                            print(f"Augmented file {augmented_file_path} still does not exist, skipping...")

        # Delete the copied original files from the augmented folder
        for file_name in os.listdir(augmented_folder_path):
            if file_name.endswith(".bin") and not re.search(r'_aug\d+\.bin$', file_name):
                file_path_to_delete = os.path.join(augmented_folder_path, file_name)
                os.remove(file_path_to_delete)

if __name__ == "__main__":
    augment_data()
