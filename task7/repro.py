import random
import numpy as np
import torch
import yaml
import logging

# -----------------------------
# Load YAML Config
# -----------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

seed = config["seed"]

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    filename=config["log_path"],
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# -----------------------------
# Reproducibility Settings
# -----------------------------
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# For GPU reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# Generate Random Tensor
# -----------------------------
a = torch.randn(3)

print("Generated Tensor:")
print(a)

logging.info(f"Generated Tensor: {a}")

# -----------------------------
# Save Checkpoint
# -----------------------------
checkpoint = {
    "seed": seed,
    "tensor": a
}

torch.save(checkpoint, config["checkpoint_path"])

print("\nCheckpoint saved.")
print("Log saved.")
