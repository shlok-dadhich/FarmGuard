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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, visualization as viz


def main():
    # Resolve repository root (the folder containing a 'configs' directory)
    ROOT = Path.cwd().resolve()
    for cand in [ROOT, *ROOT.parents]:
        if (cand / 'configs').is_dir():
            ROOT = cand
            break

    # Paths for data, outputs and model checkpoint
    DATA_ROOT = ROOT / 'data' / 'raw' / 'Cotton_data'
    OUTPUT_ROOT = ROOT / 'outputs'
    FIG_ROOT = OUTPUT_ROOT / 'figures' / 'cotton'
    LOG_ROOT = OUTPUT_ROOT / 'logs'
    METRICS_ROOT = OUTPUT_ROOT / 'metrics' / 'cotton'
    MODEL_ROOT = ROOT / 'models' / 'checkpoints' / 'cotton'

    # Ensure output directories exist
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(LOG_ROOT / 'cotton_inference.log'), logging.StreamHandler()]
    )
    logger = logging.getLogger('cotton_inference')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {DEVICE}')

    # Verify that the expected split folders exist
    SPLITS = {'train': 'train', 'val': 'val', 'test': 'test'}
    def has_splits(root):
        return all((root / d).is_dir() for d in SPLITS.values())
    if not has_splits(DATA_ROOT):
        raise FileNotFoundError(f'Data not found at {DATA_ROOT}')
    logger.info(f'Dataset root resolved: {DATA_ROOT}')

    # Transforms – keep the same as used during training
    input_size = 224
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Test dataset and loader
    test_dataset = datasets.ImageFolder(DATA_ROOT / SPLITS['test'], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load checkpoint – it contains the model state dict and optimizer state
    checkpoint_path = MODEL_ROOT / 'best_densenet.pth'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found at {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Re‑create the model architecture (DenseNet‑121) matching the training script
    num_classes = checkpoint['model_state_dict']['classifier.weight'].shape[0]
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    logger.info('Model loaded from checkpoint')

    # Evaluation on the test split
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    test_loss = loss_sum / total
    test_acc = correct / total
    logger.info(f'Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}')

    # Classification report
    class_names = test_dataset.classes
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_path = METRICS_ROOT / 'test_classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info('Classification report saved')

    # Confusion matrix visualisation
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(FIG_ROOT / 'confusion_matrix.png')
    plt.close()
    logger.info('Confusion matrix saved')

    # XAI – Integrated Gradients on a few test samples
    ig = IntegratedGradients(model)
    sample_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    os.makedirs(FIG_ROOT / 'xai', exist_ok=True)
    for idx, (img, label) in enumerate(sample_loader):
        if idx >= 5:
            break
        img = img.to(DEVICE)
        # Compute Integrated Gradients attribution and move to CPU
        attr, delta = ig.attribute(img, target=label.item(), return_convergence_delta=True)
        # Convert attribution to numpy on CPU (C,H,W)
        attr_np = attr.squeeze().cpu().detach().numpy()
        # De‑normalize image to [0,1] for proper visualisation
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original = np.transpose(img.squeeze().cpu().numpy(), (1, 2, 0))
        original = (original * std) + mean
        original = np.clip(original, 0, 1)

        # Attribution tensor (C,H,W) -> (H,W,C) for visualisation
        attr_vis = np.transpose(attr_np, (1, 2, 0))

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.title('Input')
        plt.imshow(original)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Attribution')
        try:
            viz.visualize_image_attr(attr_vis, original, method='heatmap', sign='all', show_colorbar=True, outlier_perc=1)
        except Exception:
            # Fallback: show mean absolute attribution across channels
            plt.imshow(np.mean(np.abs(attr_np), axis=0), cmap='viridis')
        plt.axis('off')
        plt.suptitle(f'True: {class_names[label.item()]}')
        plt.tight_layout()
        plt.savefig(FIG_ROOT / 'xai' / f'sample_{idx}.png')
        plt.close()
        plt.axis('off')
        plt.suptitle(f'True: {class_names[label.item()]}')
        plt.tight_layout()
        plt.savefig(FIG_ROOT / 'xai' / f'sample_{idx}.png')
        plt.close()
    logger.info('XAI visualisations saved')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
