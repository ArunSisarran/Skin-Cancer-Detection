import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

from dataset import SkinCancerDataset
from modelCNN import SkinCancerCNN
from transforms import get_transforms


def setup_data(df, img_dirs, batch_size=32):
    # Spliting data
    train_data, val_data = train_test_split(
        df, test_size=0.2, stratify=df['binary_target'], random_state=42
    )

    print(f"training: {len(train_data)}")
    print(f"valid: {len(val_data)}")
    print(f"training mel %: {train_data['binary_target'].mean():.2%}")

    # Making the datasets
    train_dataset = SkinCancerDataset(
        train_data, img_dirs, transform=get_transforms('train')
    )

    val_dataset = SkinCancerDataset(
        val_data, img_dirs, transform=get_transforms('val')
    )

    # Making the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_data


def calculate_class_weights(train_data):
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_data['binary_target']),
        y=train_data['binary_target']
    )

    return torch.FloatTensor(class_weights)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    # loading data
    df = pd.read_csv('./data/binary_metadata.csv')
    img_dirs = ['./data/HAM10000_images_part_1/', './data/HAM10000_images_part_2/']

    train_loader, val_loader, train_data = setup_data(df, img_dirs)

    # model, loss, optimizer
    model = SkinCancerCNN().to(device)

    class_weights = calculate_class_weights(train_data).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train epochs
    num_epochs = 20
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"epoch [{epoch+1}/{num_epochs}]")
        print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%")
        print(f"val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%")

        # save the best model acc we have
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best.pth')
            print(f"best model saved, val acc: {val_acc:.2f}%")

    print(f"training done, best val acc {val_acc:.2f}%")

if __name__ == "__main__":
    main()
