import os
import json
import torch

# Ensure project root is the working directory when running this script
ROOT_DIR = os.getcwd()

# Import the MTEncoder backbone and its lightweight wrapper
from src.mtencoder import MTEncoder
from src.wrap_mtencoder import MTEncoder_model


# -----------------------------------------------------------------------------
# Load configuration and build the encoder model
# -----------------------------------------------------------------------------

config_path = os.path.join(ROOT_DIR, "model_weights", "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

# Select compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the underlying MTEncoder network
encoder_network = MTEncoder(config, compute_device=device).to(device)
encoder_network.device = device  # Ensure wrapper can access the compute device

# Wrap the network with utility methods for encoding materials & load weights
mtencoder_wrapper = MTEncoder_model(
    encoder_network,
    os.path.join(ROOT_DIR, "model_weights"),
)

# Put the model in evaluation mode
mtencoder_wrapper.model.eval()

# -----------------------------------------------------------------------------
# Simple reproducibility check
# -----------------------------------------------------------------------------

test_material = ["FeNa2O4"]

output1 = mtencoder_wrapper.encode_materials(test_material)
output2 = mtencoder_wrapper.encode_materials(test_material)

print("Output 1:", output1)
print("Output 2:", output2)
print("Difference:", torch.abs(output1 - output2).sum())





