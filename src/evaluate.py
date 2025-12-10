"""
Model Evaluation Script
-----------------------
Standalone script to evaluate a trained model on the validation set.
Displays accuracy, precision, recall, F1-score, and confusion matrix.

Usage:
    python src/evaluate.py --config configs/config.yaml --model checkpoints/model_final.pth
"""

import torch
import yaml
import argparse
import matplotlib.pyplot as plt

from dataset import EmotionDataModule
from model import EmotionModel
from utils import calculate_metrics, plot_confusion_matrix


def evaluate(config_path, model_path):
    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    # Load Data
    dm = EmotionDataModule(config_path)
    _, val_loader, class_names = dm.get_dataloaders()
    
    # Load Model
    model = EmotionModel(config_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    
    # Calculate Metrics
    precision, recall, f1 = calculate_metrics(all_preds, all_labels)
    acc = (all_preds == all_labels).sum().item() / len(all_labels) * 100
    
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Show Confusion Matrix
    cm_fig = plot_confusion_matrix(all_preds, all_labels, class_names)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    evaluate(args.config, args.model)
