def count_bits(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    bit_count = len(binary_data) * 8
    return bit_count

file_path = './dataset/data/RPi1Dump/rpi1_42.bin'
bits = count_bits(file_path)
print(f'The file contains {bits} bits.')


