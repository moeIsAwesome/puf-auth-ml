import os
import shutil
import subprocess
import re

# Define the base directory where the datasets folder is located
datasets_dir = "./dataset/data"
jar_file_path = "./dataset/scripts/puf-tool.jar"
corruption_percentage = 30

# Define the corruption parameters
corruption_command = f"java -jar {jar_file_path} corrupt -b 256 -p {corruption_percentage} {{}}"

# List of PUF folders
puf_folders = ["RPi1Dump", "RPi2Dump", "RPi3Dump"]

def corrupt_data():
    for puf_folder in puf_folders:
        folder_path = os.path.join(datasets_dir, puf_folder)
        corrupted_folder_path = os.path.join(folder_path, f"{corruption_percentage}-percent-corrupted")
        os.makedirs(corrupted_folder_path, exist_ok=True)

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

                # Assuming the corruption tool generates files with a pattern like _corrBot.bin, _corrMid.bin, etc.
                # Adjust if the number of corruptions is different
                for corr_suffix in ["Bot", "Mid", "Top"]:
                    corrupted_file_name = file_name.replace(".bin", f"_corr{corr_suffix}.bin")
                    corrupted_file_path = os.path.join(corrupted_folder_path, corrupted_file_name)
                    if not os.path.exists(corrupted_file_path):
                        print(f"Corrupted file {corrupted_file_path} does not exist, retrying...")
                        time.sleep(1)  # Wait a bit and try again
                        if not os.path.exists(corrupted_file_path):
                            print(f"Corrupted file {corrupted_file_path} still does not exist, skipping...")

        # Delete the copied original files from the corrupted folder
        for file_name in os.listdir(corrupted_folder_path):
            if file_name.endswith(".bin") and not re.search(r'_corr(Bot|Mid|Top)\.bin$', file_name):
                file_path_to_delete = os.path.join(corrupted_folder_path, file_name)
                os.remove(file_path_to_delete)

if __name__ == "__main__":
    corrupt_data()
