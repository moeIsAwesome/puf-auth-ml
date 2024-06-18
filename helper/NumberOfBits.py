def count_bits(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    bit_count = len(binary_data) * 8
    return bit_count

file_path = './dataset/data/RPi2Dump/reduced_50percent/rpi2_3.bin'
bits = count_bits(file_path)
print(f'The file contains {bits} bits.')


