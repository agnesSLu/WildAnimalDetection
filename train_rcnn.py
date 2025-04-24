# CS 5330 Final Project 
# Shihua Lu, Chuhan Ren
# This is used to train rcnn model, which we have decided to discard for bad performance

import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from wildlife_coco_dataset import WildlifeCocoDataset  


def get_transform(train=True):
    transforms = [
        T.Resize((512, 512)),      
        T.ToTensor()
    ]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    print(" Starting Faster R-CNN training...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    # Load dataset
    dataset = WildlifeCocoDataset(
        image_dir='wildlife_coco/images',
        annotation_file='wildlife_coco/annotations.json',
        transforms=get_transform(train=True)
    )
    print("Total samples in dataset:", len(dataset))

    # Split dataset
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Build model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Correct number of classes (including background)
    num_classes = len(dataset.get_class_names()) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            # shift labels by +1 to reserve 0 for background class
            new_targets = []
            for t in targets:
                boxes = t['boxes'].to(device)
                labels = (t['labels'] + 1).to(device)
                image_id = t['image_id'].to(device)
                new_targets.append({'boxes': boxes, 'labels': labels, 'image_id': image_id})
            targets = new_targets

            # compute losses
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            if torch.isnan(losses):
                print(f" NaN loss at epoch {epoch+1}, skipping batch")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            ckpt = f"faster_rcnn_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f" Saved checkpoint: {ckpt}")

    # Final save
    torch.save(model.state_dict(), "faster_rcnn_wildlife_final.pt")
    print(" Training complete. Model saved as faster_rcnn_wildlife_final.pt")


if __name__ == '__main__':
    main()
