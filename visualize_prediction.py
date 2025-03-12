import subprocess
import matplotlib.pyplot as plt
import numpy as np
import struct

def run_cpp_model_predict(test_cols):
    test_cols_str = ' '.join(map(str, test_cols))
    print(f"Running model_predict with test_cols: {test_cols_str}")
    result = subprocess.run(['./model_predict.out'] + test_cols, capture_output=True, text=True)
    
    print(f"Result: {result.stdout}")
    predictions = list(map(int, result.stdout.strip().split()))
    print(f"Predictions: {predictions}")
    return predictions

def read_mnist_image(file_path, index):
    with open(file_path, 'rb') as f:
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        
        image_size = num_rows * num_cols
        offset = 16 + index * image_size
        
        f.seek(offset)
        image_data = np.frombuffer(f.read(image_size), dtype=np.uint8)
        
        return image_data.reshape(num_rows, num_cols)

def visualize_prediction(line_number, prediction):
    image = read_mnist_image("./mnist-input/t10k-images.idx3-ubyte", line_number)

    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted Digit: {prediction}')
    plt.show()

if __name__ == "__main__":
    test_number = int(input("Enter how many test cases you want to run: "))
    test_cols = []
    for i in range(test_number):
        test_cols.append(input(f"Enter the line number for test case {i+1}: "))
    predictions = run_cpp_model_predict(test_cols)
    for i in range(test_number):
        visualize_prediction(int(test_cols[i]), predictions[i])