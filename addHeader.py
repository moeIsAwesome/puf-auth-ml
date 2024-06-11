def read_binary_file(file_path, size=None):
    """Read a binary file up to a given size. If size is None, read the entire file."""
    with open(file_path, 'rb') as file:
        return file.read(size)


def append_header_to_augmented(original_file, augmented_file, output_file, header_size):
    """Append the header of the original file to the augmented file."""
    # Read the header from the original file
    header = read_binary_file(original_file, header_size)

    # Read the data from the augmented file
    augmented_data = read_binary_file(augmented_file)

    # Combine header and augmented data
    combined_data = header + augmented_data

    # Write the combined data to the output file
    with open(output_file, 'wb') as file:
        file.write(combined_data)

    print(
        f"Header from {original_file} has been appended to {augmented_file} and saved to {output_file}.")

# Define file paths


original_file = './dataset/data/RPi3Dump/rpi3_0.bin'
augmented_file = './dataset/scripts/rnd3.bin'
output_file = './rnd3.bin'
header_size = 9  # Specify the size of the header in bytes

# Append header
append_header_to_augmented(
    original_file, augmented_file, output_file, header_size)