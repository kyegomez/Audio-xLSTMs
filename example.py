import torch
from audio_xlstm.model import Spectra

# Import the necessary libraries

# Example
# Create an instance of the Spectra model with the specified parameters
model = Spectra(dim=128, depth=1, heads=1, dim_head=64, patch_size=1)

# Generate random input data of shape (1, 1024, 128)
seq_len = 32
batch_size = 4
inp_dim = 16

# Create a mock up input sequence
seq = torch.randn(seq_len, batch_size, inp_dim)

# Pass the input through the model to get the output
output = model(seq)

# Print the shape of the output tensor
print(output.shape)
