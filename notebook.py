import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

def train_model(data_dir, num_epochs=5, batch_size=32, num_classes=2, lr=1e-4):
    """
    Trains a CNN on your custom dataset using transfer learning with ResNet-18.
    
    :param data_dir: Path containing 'train' and 'val' subdirectories.
    :param num_epochs: Number of training epochs.
    :param batch_size: Mini-batch size.
    :param num_classes: Number of target classes you want to classify.
    :param lr: Learning rate.
    """
    # Define training and validation transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Normalization for pretrained models
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets for train/val
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x),
                                data_transforms[x])
        for x in ['train', 'val']
    }

    # Create data loaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=batch_size,
                                       shuffle=(x == 'train'))
        for x in ['train', 'val']
    }

    # Get class names from the dataset
    class_names = image_datasets['train'].classes
    print("Classes:", class_names)

    # Choose your device
    device = torch.device("cpu")

    # Initialize a pretrained ResNet-18 model
    model = models.resnet18(pretrained=True)
    # Replace the final FC layer to match num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward pass + optimize only in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Save the model weights
    model_path = "cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save the class names
    with open("class_names.txt", "w") as f:
        for cls in class_names:
            f.write(cls + "\n")
    print("Class names saved to class_names.txt")

if __name__ == "__main__":
    # Example usage:
    # Make sure you have a folder structure:
    # data_dir/train/<class0>/..., data_dir/train/<class1>/...
    # data_dir/val/<class0>/..., data_dir/val/<class1>/...
    train_model(data_dir="data_dir",
                num_epochs=5,
                batch_size=32,
                num_classes=6,
                lr=1e-4)
