import os
import shutil
import random
from math import floor

def split_folder(input_folder, output_folder, percentage=0.2):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    all_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Calculate the number of files to move
    num_files_to_move = floor(len(all_files) * percentage)

    # Randomly select files to move
    files_to_move = random.sample(all_files, num_files_to_move)

    # Move the selected files
    for file_name in files_to_move:
        shutil.move(os.path.join(input_folder, file_name), os.path.join(output_folder, file_name))

    print(f"Moved {len(files_to_move)} files to {output_folder}")

# Example usage:
input_folder = './dataset/data/RPi1Dump/'
output_folder = './dataset/data/test/RPi1Dump/'
split_folder(input_folder, output_folder, percentage=0.2)
