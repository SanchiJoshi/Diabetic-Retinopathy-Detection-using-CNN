import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
DATASET_PATH = "retina_dataset/colored_images"
IMG_SIZE, BATCH_SIZE = 224, 16
 
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform_val)
class_names = dataset.classes
val_size = int(0.2 * len(dataset))
_, val_set = random_split(dataset, [len(dataset) - val_size, val_size],
                          generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
 
model = models.efficientnet_v2_s(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 5)
)
model.load_state_dict(torch.load("dr_model.pt", map_location=device))
model = model.to(device).eval()
 
if __name__ == '__main__':
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
     
    y_pred, y_true = np.array(all_preds), np.array(all_labels)
     
    print("\n========= CLASSIFICATION REPORT =========\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
     
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()
     
    with open("history.json") as f: history = json.load(f)
     
    for title, tr_key, val_key in [("Accuracy", "accuracy", "val_accuracy"), ("Loss", "loss", "val_loss")]:
        plt.figure()
        plt.plot(history[tr_key], label=f"Train {title}")
        plt.plot(history[val_key], label=f"Val {title}")
        plt.title(f"{title} Curve"); plt.xlabel("Epoch"); plt.ylabel(title)
        plt.legend(); plt.tight_layout(); plt.show() 