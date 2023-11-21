import torch

# Load the checkpoint
checkpoint_path = '/home/ada/satmae/temporal/checkpoints/fmow_finetune.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

input_size = checkpoint['args'].input_size
patch_size = checkpoint['args'].patch_size


# Print the input size
print(f"Input size from checkpoint: {input_size}")
print(f"Patch size from checkpoint: {patch_size}")
