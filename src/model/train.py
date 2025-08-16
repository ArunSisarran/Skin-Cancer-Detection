import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from dataset import SkinCancerDataset
from modelCNN import SkinCancerCNN
from transforms import get_transforms
from focalLoss import get_loss_function


def setup_data(df, img_dirs, batch_size=32, augmentation_level='medium', use_melanoma_specific=False):
    train_data, val_data = train_test_split(
        df, test_size=0.2, stratify=df['binary_target'], random_state=42
    )

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Training melanoma %: {train_data['binary_target'].mean():.2%}")
    print(f"Validation melanoma %: {val_data['binary_target'].mean():.2%}")

    train_dataset = SkinCancerDataset(
        train_data, 
        img_dirs, 
        transform=get_transforms('train', img_size=224)
    )

    val_dataset = SkinCancerDataset(
        val_data, 
        img_dirs, 
        transform=get_transforms('val', img_size=224)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, train_data


def calculate_class_weights(train_data):
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_data['binary_target']),
        y=train_data['binary_target']
    )
    return torch.FloatTensor(class_weights)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, (images, labels) in enumerate(pbar):
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
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    melanoma_indices = np.where(np.array(all_labels) == 1)[0]
    if len(melanoma_indices) > 0:
        melanoma_recall = recall_score(all_labels, all_preds, pos_label=1)
        melanoma_precision = precision_score(all_labels, all_preds, pos_label=1)
    else:
        melanoma_recall = melanoma_precision = 0.0

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'melanoma_recall': melanoma_recall,
        'melanoma_precision': melanoma_precision
    }


def validate_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation')

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    melanoma_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    melanoma_precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    melanoma_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    
    probs_array = np.array(all_probs)
    max_probs = np.max(probs_array, axis=1)
    confident_threshold = 0.8
    confident_wrong = (max_probs > confident_threshold) & (np.array(all_preds) != np.array(all_labels))
    confident_wrong_melanoma = (np.array(all_labels) == 1) & (np.array(all_preds) == 0) & (max_probs > confident_threshold)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'melanoma_recall': melanoma_recall,
        'melanoma_precision': melanoma_precision,
        'melanoma_f1': melanoma_f1,
        'confident_wrong_total': confident_wrong.sum(),
        'confident_wrong_melanoma': confident_wrong_melanoma.sum(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }


def print_detailed_metrics(train_metrics, val_metrics, epoch):
    print(f"\n=== Epoch {epoch+1} Results ===")
    print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
    print(f"Train - Melanoma Recall: {train_metrics['melanoma_recall']:.4f}, Precision: {train_metrics['melanoma_precision']:.4f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
    print(f"Val   - Melanoma Recall: {val_metrics['melanoma_recall']:.4f}, Precision: {val_metrics['melanoma_precision']:.4f}")
    print(f"Val   - Melanoma F1: {val_metrics['melanoma_f1']:.4f}")
    print(f"Val   - Confident Wrong (Total): {val_metrics['confident_wrong_total']}")
    print(f"Val   - Confident Wrong (Melanoma): {val_metrics['confident_wrong_melanoma']}")
    print("Confusion Matrix (Val):")
    print(val_metrics['confusion_matrix'])
    print("-" * 50)


def save_training_plots(train_history, val_history, save_path='training_plots.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    axes[0, 0].plot(epochs, train_history['loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_history['loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, train_history['accuracy'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, val_history['accuracy'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(epochs, train_history['melanoma_recall'], 'b-', label='Train Recall')
    axes[0, 2].plot(epochs, val_history['melanoma_recall'], 'r-', label='Val Recall')
    axes[0, 2].set_title('Melanoma Recall (Sensitivity)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(epochs, train_history['melanoma_precision'], 'b-', label='Train Precision')
    axes[1, 0].plot(epochs, val_history['melanoma_precision'], 'r-', label='Val Precision')
    axes[1, 0].set_title('Melanoma Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, val_history['melanoma_f1'], 'g-', label='Val F1')
    axes[1, 1].set_title('Melanoma F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(epochs, val_history['confident_wrong_melanoma'], 'r-', label='Confident Wrong Melanoma')
    axes[1, 2].set_title('Confident Wrong Melanoma Predictions')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Current working directory: {os.getcwd()}")

    data_path = './data/combined_balanced_metadata.csv'
    
    img_dirs = [
    './data/HAM10000_images_part_1/',
    './data/HAM10000_images_part_2/',
    './data/ham10000_images_part_1',
    './data/ham10000_images_part_2',
    './data/ISIC_2019_Training_Input/',
    './data/train/',  # ISIC 2020 images
    ]
    
    existing_dirs = []
    for img_dir in img_dirs:
        if os.path.exists(img_dir):
            existing_dirs.append(img_dir)
            print(f"✓ Found image directory: {img_dir}")
        else:
            print(f"✗ Directory not found: {img_dir}")
    
    if not os.path.exists(data_path):
        print(f"❌ Could not find metadata file: {data_path}")
        return
    
    if len(existing_dirs) == 0:
        print("❌ No image directories found!")
        return
    
    print(f"✓ Using {len(existing_dirs)} image directories")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    if 'binary_target' not in df.columns:
        print("Creating binary_target column...")
        df['binary_target'] = (df['dx'] == 'mel').astype(int)
    
    print(f"Dataset size: {len(df)}")
    print(f"Melanoma cases: {df['binary_target'].sum()} ({df['binary_target'].mean():.2%})")

    config = {
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 1e-3,
        'augmentation_level': 'light',  
        'use_melanoma_specific': True,  
        'loss_type': 'melanoma_focused',  
        'early_stopping_patience': 10,
        'lr_scheduler_patience': 5,
        'model_save_metric': 'melanoma_recall'  
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    train_loader, val_loader, train_data = setup_data(
        df, existing_dirs, 
        batch_size=config['batch_size'],
        augmentation_level=config['augmentation_level'],
        use_melanoma_specific=config['use_melanoma_specific']
    )

    model = SkinCancerCNN().to(device)
    
    class_weights = calculate_class_weights(train_data).to(device)
    print(f"Class weights: {class_weights}")

    if config['loss_type'] == 'focal':
        criterion = get_loss_function('focal', gamma=2.0, class_weights=class_weights)
    elif config['loss_type'] == 'melanoma_focused':
        criterion = get_loss_function('melanoma_focused', melanoma_weight=3.0, confidence_penalty=0.3)
    elif config['loss_type'] == 'asymmetric':
        criterion = get_loss_function('asymmetric', gamma_neg=4, gamma_pos=1)
    else:
        criterion = get_loss_function('weighted_ce', class_weights=class_weights)
    
    print(f"Using loss function: {config['loss_type']}")

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=config['lr_scheduler_patience']
    )

    train_history = {
        'loss': [], 'accuracy': [], 'melanoma_recall': [], 'melanoma_precision': []
    }
    val_history = {
        'loss': [], 'accuracy': [], 'melanoma_recall': [], 'melanoma_precision': [], 
        'melanoma_f1': [], 'confident_wrong_total': [], 'confident_wrong_melanoma': []
    }
    
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("=" * 60)

    for epoch in range(config['num_epochs']):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        for key in train_history.keys():
            train_history[key].append(train_metrics[key])
        for key in val_history.keys():
            val_history[key].append(val_metrics[key])
        
        print_detailed_metrics(train_metrics, val_metrics, epoch)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics[config['model_save_metric']])
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"📉 Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
        
        current_metric = val_metrics[config['model_save_metric']]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'config': config,
                'val_metrics': val_metrics
            }, 'best_model_focal.pth')
            
            print(f"🎯 New best {config['model_save_metric']}: {best_metric:.4f} (saved model)")
        else:
            patience_counter += 1
            
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        print()

    print("=" * 60)
    print("Training completed!")
    print(f"Best {config['model_save_metric']}: {best_metric:.4f} at epoch {best_epoch + 1}")
    
    save_training_plots(train_history, val_history, 'training_progress.png')
    
    checkpoint = torch.load('best_model_focal.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nFinal evaluation on validation set:")
    final_metrics = validate_epoch(model, val_loader, criterion, device, -1)
    
    print(f"Final Validation Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Final Melanoma Recall (Sensitivity): {final_metrics['melanoma_recall']:.4f}")
    print(f"Final Melanoma Precision: {final_metrics['melanoma_precision']:.4f}")
    print(f"Final Melanoma F1: {final_metrics['melanoma_f1']:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(
        final_metrics['all_labels'], 
        final_metrics['all_preds'],
        target_names=['Non-Melanoma', 'Melanoma']
    ))


if __name__ == "__main__":
    main()
