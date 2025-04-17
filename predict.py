import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
from data_loader import get_dataloaders
import torch.nn as nn

# Class names (in order from ImageFolder)
class_names = ['cat', 'dog', 'wild']

def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))  # CHW → HWC
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

def load_model(device):
    model = models.mobilenet_v2(weights=None)  # no pre-trained weights now
    model.classifier[1] = nn.Linear(model.last_channel, 3)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_show(num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    _, val_loader = get_dataloaders()

    images_shown = 0
    plt.figure(figsize=(10, 5))

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for i in range(inputs.size(0)):
            if images_shown >= num_images:
                break

            plt.subplot(1, num_images, images_shown + 1)
            title = f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}"
            imshow(inputs[i].cpu(), title)
            images_shown += 1

        if images_shown >= num_images:
            break

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_and_show(num_images=3)