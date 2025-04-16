from data_loader import get_dataloaders

def main():
    train_loader, val_loader = get_dataloaders()

    # Test: print one batch
    for images, labels in train_loader:
        print("Batch of images:", images.shape)
        print("Batch of labels:", labels)
        break  # only one batch for demo

if __name__ == "__main__":
    main()