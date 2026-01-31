from fastapi import FastAPI, UploadFile, File
from src.models.model import ParkinsonAlexNet
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
import io

app = FastAPI(title="Parkinson's AI Diagnostic System")

# 1. Load the Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParkinsonAlexNet(num_classes=2)

# Load the weights you just trained
# ensure 'trained_weights/parkinson_model.pth' exists from your training step
model.load_state_dict(torch.load("trained_weights/parkinson_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# 2. Define the exact same transforms used in training
def transform_image(image_bytes):
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Define the transformation pipeline (Must match training!)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), # AlexNet standard input size
        transforms.ToTensor(),         # Convert to Tensor (0-1 range)
    ])
    
    # Apply transform and add a "batch" dimension (1, 3, 224, 224)
    return preprocess(image).unsqueeze(0).to(device)

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    # A. Read and Preprocess the Image
    contents = await file.read()
    input_tensor = transform_image(contents)
    
    # B. Run Model Inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Convert raw output to probabilities (percentages)
        probabilities = F.softmax(output, dim=1)
        
        # Index 0 = Control (Healthy), Index 1 = PD (Parkinson's)
        # We get the probability of Parkinson's (Index 1)
        pd_probability = probabilities[0][1].item() * 100
        
    # C. Interpret Results
    if pd_probability > 50:
        diagnosis = "Parkinson's Detected"
        status = "Positive"
    else:
        diagnosis = "Healthy / Control"
        status = "Negative"

    # D. Mock RAG (Keep this mocked until you add Pinecone keys)
    # This prevents the app from crashing while you focus on the model.
    similar_cases = [
        {"id": "DB-291", "outcome": "Similar feature patterns found in Stage 1 PD."},
        {"id": "DB-310", "outcome": "Patient responded well to early intervention."}
    ]
    report = f"AI Analysis: The model detected {status} signs with {pd_probability:.2f}% confidence."

    return {
        "diagnosis": diagnosis,
        "confidence": f"{pd_probability:.2f}%",
        "ai_report": report,
        "similar_cases": similar_cases if status == "Positive" else []
    }