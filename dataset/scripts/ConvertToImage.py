import os
import subprocess
import shutil

# Define the folder containing the .bin files and the output folder for images
input_folder = './dataset/data/RPi1Dump/'
height = 4096
width = 2048
output_folder = f'./dataset/data/RPi1Dump/images/{height}-{width}'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the Java command template
java_command_template = f'java -jar ./dataset/scripts/puf-tool.jar image -f=bin -H={height} -w={width} {{}} -c'

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
        generated_image_name = file_name.replace('.bin', '.png')
        generated_image_path = os.path.join(input_folder, generated_image_name)
        if os.path.exists(generated_image_path):
            shutil.move(generated_image_path, output_file_path)

print("Conversion of all .bin files to images is complete.")
