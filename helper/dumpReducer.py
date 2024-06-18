import os

def reduce_bin_file_size(file_path, output_folder, keep_bits):
    keep_bytes = keep_bits // 8  # Convert bits to bytes

    # Read the original binary file
    with open(file_path, 'rb') as f:
        data = f.read()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Write the truncated data to a new file in the output folder
    filename = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, filename)
    with open(output_file_path, 'wb') as f:
        f.write(data[:keep_bytes])

def process_folder(input_folder, keep_bits):
    # Define the output folder path
    output_folder = os.path.join(input_folder, f"reduced_to_{keep_bits}bits")
    # Iterate over all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".bin"):
            file_path = os.path.join(input_folder, filename)
            reduce_bin_file_size(file_path, output_folder, keep_bits)
            print(f"Processed {filename}")

if __name__ == "__main__":
    # Set the folder path and number of bits to keep here
    input_folder = "./dataset/data/RPi1Dump/"  # Replace with the actual folder path
    keep_bits = 16  # Replace with the desired number of bits to keep

    process_folder(input_folder, keep_bits)
    print("Processing complete.")
