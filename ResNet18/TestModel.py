# TestModel.py
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Paths
MODEL_PATH = Path("models/resnet18_colormap/best.pth")
DATA_DIR = Path("Data/colormap_truncate")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1️⃣ Load the checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = len(checkpoint["class_to_idx"])
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 2️⃣ Rebuild the same model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()
print(f"Loaded model with {num_classes} classes.")

# 3️⃣ Define transforms (same as validation)
transform = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# 4️⃣ Load dataset for evaluation
dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 5️⃣ Evaluate accuracy
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in tqdm(loader, desc="Testing"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = 100.0 * correct / total
print(f"✅ Overall Accuracy: {acc:.2f}%")

# Optional: test a single image
def predict_one(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out, 1)
    print(f"Prediction for {Path(img_path).name}: {idx_to_class[pred.item()]}")

# Example:
# predict_one("Data/colormap_truncate/class3/some_image.png")
