import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "retina_dataset/colored_images"
IMG_SIZE, BATCH_SIZE, EPOCHS_HEAD, EPOCHS_FINE = 224, 48, 10, 15

# Added ColorJitter and GaussianBlur to reduce overfitting
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform_train)
val_size = int(0.2 * len(dataset))
train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size],
                                  generator=torch.Generator().manual_seed(42))
val_set.dataset.transform = transform_val

train_loader = None
val_loader = None

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
for p in model.parameters(): p.requires_grad = False

# Increased Dropout 0.5 → 0.6 to reduce overfitting
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 128), nn.ReLU(), nn.Dropout(0.6), nn.Linear(128, 5)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.5, 0.7, 0.3, 2.8, 3.5]).to(device))

def train(epochs, optimizer):
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=3)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_loss, wait = float("inf"), 0

    for epoch in range(1, epochs + 1):
        model.train()
        tl, tc, tt = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(x); loss = criterion(out, y)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            tl += loss.item() * x.size(0); tc += (out.argmax(1) == y).sum().item(); tt += x.size(0)

        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    vl += criterion(out, y).item() * x.size(0); vc += (out.argmax(1) == y).sum().item(); vt += x.size(0)

        tl, ta, vl, va = tl/tt, tc/tt, vl/vt, vc/vt
        for k, v in zip(history, [tl, ta, vl, va]): history[k].append(v)
        scheduler.step(vl)
        print(f"Epoch {epoch:02d} | Loss {tl:.4f} Acc {ta:.4f} | Val Loss {vl:.4f} Val Acc {va:.4f}")

        if vl < best_loss: best_loss, wait = vl, 0; torch.save(model.state_dict(), "best_weights.pt")
        else:
            wait += 1
            if wait >= 3: print("Early stopping."); break   # patience 5 → 3

    model.load_state_dict(torch.load("best_weights.pt"))
    return history

if __name__ == '__main__':
    print(f"Using: {device}")
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=(device.type == "cuda"))

    h1 = train(EPOCHS_HEAD, optim.Adam(model.classifier.parameters(), lr=1e-4))

    # Unfreezing last 5 blocks instead of 10 to reduce overfitting
    for layer in list(model.features.children())[-5:]:
        for p in layer.parameters(): p.requires_grad = True

    # Added weight_decay for L2 regularization
    h2 = train(EPOCHS_FINE, optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4))

    torch.save(model.state_dict(), "dr_model.pt")
    history = {k: [float(x) for x in h1[k] + h2[k]] for k in h1}
    with open("history.json", "w") as f: json.dump(history, f)
    print("Done. Saved dr_model.pt")