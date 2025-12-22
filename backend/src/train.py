import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import json
from datetime import datetime

# DATASET
class PreExtractedFeaturesDataset(Dataset):
    """Dataset that loads pre-extracted features"""
    
    def __init__(self, features_file):
        print(f"Loading features from {features_file}...")
        
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        data = np.load(features_file)
        
        self.features = torch.from_numpy(data['features']).float()
        self.labels = torch.from_numpy(data['labels']).long()
        
        
        print(f"  ✓ Loaded {len(self.features)} samples")
        print(f"  ✓ Feature shape: {self.features.shape}")
        print(f"  ✓ Label distribution: {self.labels.sum().item()} fake, {(self.labels == 0).sum().item()} real")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# MODEL
class Classifier(nn.Module):
    """Simple classifier for pre-extracted features"""
    
    def __init__(self, input_dim=775, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, features):
        return self.classifier(features)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# EVALUATION
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
    except:
        auc = 0.0
        ap = 0.0
    
    return avg_loss, accuracy, auc, ap

# TRAINING
def train():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    # model is really powerful so 1 epoch is enough to get good results and avoid overfitting
    num_epochs = 1 

    learning_rate = 1e-4
    hidden_dim = 256
    dropout = 0.3
    
    print("="*60)
    print("TRAINING AI IMAGE DETECTOR")
    print("="*60)
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_dataset = PreExtractedFeaturesDataset('data/features/train_features.npz')
    eval_dataset = PreExtractedFeaturesDataset('data/features/eval_features.npz')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Eval batches: {len(eval_loader)}")
    
    # Create model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = Classifier(
        input_dim=775,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    print(f"Trainable parameters: {model.get_num_params():,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )'''
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_ap': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward
            logits = model(features)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation phase
        val_loss, val_acc, val_auc, val_ap = evaluate(
            model, eval_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_ap'].append(val_ap)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        #scheduler.step(val_acc)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc:.4f}, AP={val_ap:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_ap': val_ap,
                'train_acc': train_acc,
                'config': {
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                    'learning_rate': learning_rate
                }
            }, 'checkpoints/best_model.pth')
            
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print("-" * 60)
    
    # Training complete
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final Validation AUC: {history['val_auc'][-1]:.4f}")
    print(f"Final Validation AP: {history['val_ap'][-1]:.4f}")
    
    # Save training history
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\n✓ Saved training history to results/training_history.json")
    
    # Plot training curves
    plot_training_curves(history, num_epochs)
    
    # Save final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': {
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'learning_rate': learning_rate
        }
    }, 'checkpoints/final_model.pth')
    print("✓ Saved final model to checkpoints/final_model.pth")
    
    return model, history

# PLOTTING
def plot_training_curves(history, num_epochs):
    """Plot and save training curves"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, num_epochs + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(epochs, history['val_auc'], 'g-o', label='Validation AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('AUC-ROC', fontsize=12)
    axes[1, 0].set_title('Validation AUC-ROC', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Average Precision
    axes[1, 1].plot(epochs, history['val_ap'], 'm-o', label='Validation AP', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Average Precision', fontsize=12)
    axes[1, 1].set_title('Validation Average Precision', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training curves to results/training_curves.png")
    
    # Show plot
    plt.show()

def main():
    start_time = datetime.now()
    
    try:
        model, history = train()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"Total training time: {duration}")
        print(f"{'='*60}")
        
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()