import torch
import torch.nn as nn
import torchvision.models as models

class ParkinsonAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ParkinsonAlexNet, self).__init__()
        # Load standard AlexNet
        self.alexnet = models.alexnet(weights='DEFAULT')
        
        # Modify the classifier for 2 classes (PD vs Control)
        # Using Sigmoid logic as per your original code (usually handled by CrossEntropy, but adapted here)
        self.alexnet.classifier[6] = nn.Sequential(
            nn.Linear(4096, num_classes),
            # Note: For training with CrossEntropyLoss, usually we remove Softmax/Sigmoid here
            # But if you want probabilities directly:
            nn.Softmax(dim=1) 
        )
        
        # Feature extractor wrapper for RAG (getting embeddings)
        self.feature_extractor = nn.Sequential(
            *list(self.alexnet.features.children()),
            self.alexnet.avgpool,
            nn.Flatten(),
            *list(self.alexnet.classifier.children())[:-1] # Stop before final layer
        )

    def forward(self, x):
        return self.alexnet(x)

    def get_embedding(self, x):
        """
        Extracts the 4096-dim vector for RAG similarity search
        """
        with torch.no_grad():
            return self.feature_extractor(x)