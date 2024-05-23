import os
import subprocess
import shutil

# Define the folder containing the .bin files and the output folder for images
input_folder = './datasets/dataset_ready/RPi3Dump'
output_folder = './datasets/dataset_ready/datasets_images/RPi3Dump'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the Java command template
java_command_template = 'java -jar ./datasets/pufmetrics-1.0-SNAPSHOT.jar image -f=bin -H=4096 -w=2048 {} -c'

# Iterate over all files in the input folder
for file_name in os.listdir(input_folder):
    # Check if the file is a .bin file
    if file_name.endswith('.bin'):
        # Construct the full file path
        input_file_path = os.path.join(input_folder, file_name)

        # Construct the output file name
        # Assuming the output images are PNGs
        output_file_name = file_name.replace('.bin', '.png')
        output_file_path = os.path.join(output_folder, output_file_name)

        # Construct the full Java command
        java_command = java_command_template.format(input_file_path)

        # Execute the command
        subprocess.run(java_command, shell=True, check=True)

        # Move the generated image to the output folder
        # Adjust based on how the tool names the output
        generated_image_name = input_file_path.replace('.bin', '.png')
        if os.path.exists(generated_image_name):
            shutil.move(generated_image_name, output_file_path)

print("Conversion of all .bin files to images is complete.")
