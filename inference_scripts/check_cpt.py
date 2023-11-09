

import torch

# Load the checkpoint
checkpoint_path = '/home/ada/satmae/multispectral/checkpoints/finetune-vit-large-e7.pth'  # Replace with the actual path to your checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Retrieve the input size from the checkpoint
input_size = checkpoint['args'].input_size

# Print the input size
print(f"Input size from checkpoint: {input_size}")
