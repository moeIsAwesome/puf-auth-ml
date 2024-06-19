import os
import subprocess

def generate_random_binary_files(jar_path, bits, num_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_files):
        output_file = os.path.join(output_dir, f"rnd{i}.bin")
        command = f"java -jar {jar_path} random -b={bits} {output_file}"
        try:
            print(f"Executing command: {command}")
            subprocess.run(command, shell=True, check=True)
            print(f"Generated file: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating file {output_file}: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")

# Set the parameters
jar_path = './dataset/scripts/puf-tool.jar'  # Path to your JAR file
bits = 8388608  # Number of bits to generate
num_files = 8  # Number of random files to generate
output_dir = f'./dataset/random/rnd_{bits}'  # Directory to save the generated files

# Generate the random binary files
generate_random_binary_files(jar_path, bits, num_files, output_dir)
