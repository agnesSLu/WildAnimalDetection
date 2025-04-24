# CS 5330 Final Project 
# Shihua Lu, Chuhan Ren
# This is used to test ResNet model

import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import os
import csv

# 1. Setup device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# 2. Define your classes in the same order as during training
classes = ['buffalo', 'elephant', 'rhino', 'zebra']  

# 3. Build model architecture and load weights
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("four_classes_resnet_classifier_wildlife.pt", map_location=device))
model.to(device)
model.eval()

# 4. Define inference transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 5. Inference on a single image
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    return classes[pred_idx], probs[pred_idx].item()

# 6. Run on all images in a folder and write to CSV
test_folder = "wildlife_coco/images"  # your folder here
output_file = "predictions.csv"

with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'predicted_class', 'confidence', 'isMeet'])

    for fname in os.listdir(test_folder):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(test_folder, fname)
        cls, score = predict_image(path)
        writer.writerow([fname, cls, f"{score:.4f}", f"{fname.split('_')[0] == cls}"])

print(f" Predictions written to {output_file}")