import sys
import os

# --- PATH FIX ---
# Force Python to find the 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms  # <--- Added this
from src.data.dataset import ParkinsonDataset
from src.models.model import ParkinsonAlexNet

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10
DATA_PATH = "images"
SAVE_PATH = "trained_weights/parkinson_model.pth"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # --- THE FIX: Define Resizing Transform ---
    # AlexNet expects 224x224 inputs. 
    # Since your dataset returns Numpy arrays, we convert to PIL -> Resize -> Tensor.
    data_transform = transforms.Compose([
        transforms.ToPILImage(),       # Convert Numpy array to PIL Image
        transforms.Resize((224, 224)), # Force resize to 224x224
        transforms.ToTensor()          # Convert to PyTorch Tensor
    ])

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data folder '{DATA_PATH}' not found!")
        return

    print("📂 Loading dataset...")
    # Pass the transform here so images get resized automatically
    full_dataset = ParkinsonDataset(root_dir=DATA_PATH, enhance=True, transform=data_transform)
    
    # Check if dataset is empty
    if len(full_dataset) == 0:
        print("❌ Error: No images found. Check your 'images/PD' and 'images/Control' folders.")
        return

    # Split: 80% Training, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    print(f"✅ Data Loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

    # Initialize Model
    model = ParkinsonAlexNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🧠 Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save Model
    os.makedirs("trained_weights", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 Model saved successfully to {SAVE_PATH}")

if __name__ == "__main__":
    train()