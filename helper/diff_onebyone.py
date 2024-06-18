def compare_files(file1_path, file2_path):
    # Read the binary data from both files
    with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
        data1 = file1.read()
        data2 = file2.read()

    # Ensure both files are of the same length
    if len(data1) != len(data2):
        raise ValueError(
            "Files are of different lengths and cannot be directly compared.")

    # Count the number of differing bytes
    differing_bytes = sum(b1 != b2 for b1, b2 in zip(data1, data2))

    # Calculate the percentage difference
    total_bytes = len(data1)
    percentage_difference = (differing_bytes / total_bytes) * 100

    return percentage_difference


# File paths
file1_path = './dataset/data/RPi2Dump/rpi2_3.bin'
file2_path = './dataset/data/RPi2Dump/8192-64-augmented/rpi2_3_aug1.bin'

# Compare the files and print the percentage difference
percentage_difference = compare_files(file1_path, file2_path)
print(f"The files differ by {percentage_difference:.2f}%")
