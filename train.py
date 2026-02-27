import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder("train", transform=train_transform)
val_dataset = datasets.ImageFolder("val", transform=val_test_transform)
test_dataset = datasets.ImageFolder("test", transform=val_test_transform)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load pretrained MobileNetV2
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier for binary classification
model.classifier[1] = nn.Linear(model.last_channel, 1)
model = model.to(device)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 8
best_accuracy = 0.0

# ------------------- TRAINING -------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    print(f"\nEpoch [{epoch+1}/{epochs}] Starting...")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}] Training Loss: {avg_loss:.4f}")

    # ------------------- VALIDATION -------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            predicted = (predictions > 0.5).int()

            correct += (predicted.squeeze() == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), "mydoc_model.pth")
        print("Best model saved!")

# ------------------- TEST EVALUATION -------------------
model.eval()
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.sigmoid(outputs)
        predicted = (predictions > 0.5).int()

        correct += (predicted.squeeze() == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.squeeze().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

# ------------------- CONFUSION MATRIX -------------------
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                            target_names=["NORMAL", "PNEUMONIA"]))

print("\nTraining Finished!")