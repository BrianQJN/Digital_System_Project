import Encoder
import Decoder
import numpy as np

from PIL import Image

# Import necessary modules and classes

# Load the sample image
sample_image_path = 'A1/images/000.png'

# Initialize Encoder and Decoder objects
encoder = Encoder()
decoder = Decoder()

# Define parameters
i = 8  # Block size
QP = 6  # Quantization Parameter

# Load the sample image
original_image = Image.open(sample_image_path)

# Convert the image to a numpy array
image_array = np.array(original_image)

# Get the dimensions of the image
height, width, _ = image_array.shape

# Initialize an empty array for the reconstructed image
reconstructed_image = np.zeros_like(image_array)

# Loop through the image in (𝑖 × 𝑖) blocks
for y in range(0, height, i):
    for x in range(0, width, i):
        # Extract the current (𝑖 × 𝑖) block
        block = image_array[y:y+i, x:x+i]

        # Apply 2D DCT transform and quantization using Encoder
        transformed_residual_block = encoder.apply_dct_to_residual_block(block)
        quantized_coefficients = encoder.quantize(transformed_residual_block, QP)

        # Inverse quantization and inverse 2D DCT transform using Decoder
        coefficients = decoder.inverse_quantize_coefficients(quantized_coefficients, QP)
        reconstructed_block = decoder.apply_inverse_dct_to_coefficients(coefficients)

        # Place the reconstructed block into the reconstructed image
        reconstructed_image[y:y+i, x:x+i] = reconstructed_block

# Convert the numpy array back to an image
reconstructed_image = Image.fromarray(reconstructed_image.astype('uint8'))

# Save the reconstructed image
reconstructed_image.save('reconstructed_image.png')

# Display or further analyze the reconstructed image
reconstructed_image.show()
