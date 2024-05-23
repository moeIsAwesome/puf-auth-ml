import os


def compare_files(file1_path, file2_path):
    # Read the binary data from both files
    with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
        data1 = file1.read()
        data2 = file2.read()

    # Ensure both files are of the same length
    if len(data1) != len(data2):
        raise ValueError(
            f"Files {file1_path} and {file2_path} are of different lengths and cannot be directly compared.")

    # Count the number of differing bytes
    differing_bytes = sum(b1 != b2 for b1, b2 in zip(data1, data2))

    # Calculate the percentage difference
    total_bytes = len(data1)
    percentage_difference = (differing_bytes / total_bytes) * 100

    return percentage_difference


def main():
    # Directories containing the files
    directory1 = './datasets/RPi1Dump/corrupted'
    directory2 = './datasets/RPi1Dump'

    # Get all files that match the naming pattern in both directories
    file_pattern1 = 'rpi1_'
    file_pattern2 = 'rpi1_'

    files1 = [f for f in os.listdir(directory1) if f.startswith(
        file_pattern1) and f.endswith('.bin')]
    files2 = [f for f in os.listdir(directory2) if f.startswith(
        file_pattern2) and f.endswith('.bin')]

    # Sort the files to ensure they are in order
    files1.sort()
    files2.sort()

    # List to store the differences
    differences = []

    # Compare each pair of files from directory1 and directory2
    for file1 in files1:
        for file2 in files2:
            file1_path = os.path.join(directory1, file1)
            file2_path = os.path.join(directory2, file2)
            try:
                percentage_difference = compare_files(file1_path, file2_path)
                differences.append((file1, file2, percentage_difference))
                print(
                    f"The files {file1} and {file2} differ by {percentage_difference:.2f}%")
            except ValueError as e:
                print(e)

    # Find the highest and lowest differences
    if differences:
        highest_diff = max(differences, key=lambda x: x[2])
        lowest_diff = min(differences, key=lambda x: x[2])

        print(
            f"\nThe highest difference is between {highest_diff[0]} and {highest_diff[1]}: {highest_diff[2]:.2f}%")
        print(
            f"The lowest difference is between {lowest_diff[0]} and {lowest_diff[1]}: {lowest_diff[2]:.2f}%")


if __name__ == "__main__":
    main()
