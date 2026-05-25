import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Mild","Moderate","No_DR","Proliferate_DR","Severe"]

model = models.efficientnet_v2_s(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features,128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128,5)
)
model.load_state_dict(torch.load("dr_model.pt", map_location=device))
model=model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(img_path):
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1).squeeze()
    print(f"Prediction: {CLASS_NAMES[probs.argmax()]}")
    print(f"Confidence: {probs.max():.4f}")
 
predict_image("severe_test.png")

