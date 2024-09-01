import torch
from audio_xlstm.model import Spectra

# Import the necessary libraries

# Example
# Create an instance of the Spectra model with the specified parameters
model = Spectra(dim=128, depth=1, heads=1, dim_head=64, patch_size=16)

# Generate random input data of shape (1, 1024, 128)
input = torch.randn(1, 1024, 128)

# Pass the input through the model to get the output
output = model(input)

# Print the shape of the output tensor
print(output.shape)
