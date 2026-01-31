import torch
import os
from src.models.model import ParkinsonAlexNet

# 1. Initialize the model (random weights)
model = ParkinsonAlexNet()

# 2. Create directory if it doesn't exist
os.makedirs("trained_weights", exist_ok=True)

# 3. Save the weights
torch.save(model.state_dict(), "trained_weights/parkinson_model.pth")
print("✅ Success! Created 'trained_weights/parkinson_model.pth'")