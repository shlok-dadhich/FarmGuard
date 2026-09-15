# Imports
import os
import json
import logging
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import datasets, models, transforms
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from captum.attr import IntegratedGradients, visualization as viz


def main():
    # Repository root discovery
    ROOT = Path.cwd().resolve()
    for cand in [ROOT, *ROOT.parents]:
        if (cand / 'configs').is_dir():
            ROOT = cand
            break

    DATA_ROOT = ROOT / 'data' / 'raw' / 'Cotton_data'
    OUTPUT_ROOT = ROOT / 'outputs'
    FIG_ROOT = OUTPUT_ROOT / 'figures' / 'cotton'
    LOG_ROOT = OUTPUT_ROOT / 'logs'
    METRICS_ROOT = OUTPUT_ROOT / 'metrics' / 'cotton'
    MODEL_ROOT = ROOT / 'models' / 'checkpoints' / 'cotton'

    # Ensure directories exist
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.FileHandler(LOG_ROOT/'cotton_training.log'), logging.StreamHandler()])
    logger = logging.getLogger('cotton_training')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {DEVICE}')

    # Verify dataset splits
    SPLITS = {'train': 'train', 'val': 'val', 'test': 'test'}
    def has_splits(root):
        return all((root / d).is_dir() for d in SPLITS.values())
    if not has_splits(DATA_ROOT):
        raise FileNotFoundError(f'Data not found at {DATA_ROOT}')
    logger.info(f'Dataset root resolved: {DATA_ROOT}')

    # Transforms
    input_size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(DATA_ROOT / SPLITS['train'], transform=train_transform)
    val_dataset   = datasets.ImageFolder(DATA_ROOT / SPLITS['val'],   transform=val_transform)
    test_dataset  = datasets.ImageFolder(DATA_ROOT / SPLITS['test'],  transform=val_transform)
    class_names = train_dataset.classes
    logger.info(f'Classes: {class_names}')

    # Weighted sampler for class imbalance
    class_counts = np.bincount([s[1] for s in train_dataset.samples])
    class_weights = 1.0 / class_counts
    samples_weights = [class_weights[s[1]] for s in train_dataset.samples]
    sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model – DenseNet121
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))
    model = model.to(DEVICE)
    logger.info(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, all_labels, all_preds

    # Training loop with early stopping
    num_epochs = 40
    best_val_acc = 0.0
    patience = 10
    early_stop_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        logger.info(f'Epoch {epoch}/{num_epochs} – Train loss: {train_loss:.4f} acc: {train_acc:.4f} – Val loss: {val_loss:.4f} acc: {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, MODEL_ROOT / 'best_densenet.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info('Early stopping triggered')
                break

    # Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(FIG_ROOT / 'training_curves.png')
    plt.close()

    # Load best model and evaluate on test set
    checkpoint = torch.load(MODEL_ROOT / 'best_densenet.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)
    logger.info(f'Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}')
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_path = METRICS_ROOT / 'test_classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(FIG_ROOT / 'confusion_matrix.png')
    plt.close()

    # XAI – Integrated Gradients on a few test samples
    ig = IntegratedGradients(model)
    sample_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    os.makedirs(FIG_ROOT / 'xai', exist_ok=True)
    for idx, (img, label) in enumerate(sample_loader):
        if idx >= 5:
            break
        img = img.to(DEVICE)
        attr, delta = ig.attribute(img, target=label.item(), return_convergence_delta=True)
        attr = attr.squeeze().cpu().detach().numpy()
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.title('Input')
        plt.imshow(np.transpose(img.squeeze().cpu().numpy(), (1,2,0)))
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title('Attribution')
        viz.visualize_image_attr(attr, np.transpose(img.squeeze().cpu().numpy(), (1,2,0)), method='heatmap', sign='all', show_colorbar=True, outlier_perc=1)
        plt.axis('off')
        plt.suptitle(f'True: {class_names[label.item()]}')
        plt.tight_layout()
        plt.savefig(FIG_ROOT / 'xai' / f'sample_{idx}.png')
        plt.close()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
