"""
Utility Functions
-----------------
Common helper functions for:
- Metric calculation (Precision, Recall, F1)
- Visualization (Confusion Matrix, plots)
- Educational text generation for the dashboard
- Loss functions (SoftTargetCrossEntropy) and Mixup logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import io
import PIL.Image
from torchvision.transforms import ToTensor

def calculate_metrics(preds, labels, average='weighted'):
    """
    Calculate Precision, Recall, F1-Score.
    """
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0
    )
    return precision, recall, f1

def plot_confusion_matrix(preds, labels, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    cm = confusion_matrix(labels, preds)
    
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and cannot be used after this call.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image

def get_educational_text():
    """
    Returns the educational text for the dashboard.
    """
    text = """
    ### Metric Guide
    
    **1. Cross Entropy Loss**
    - **What it is:** Measures the difference between the predicted probability distribution and the actual distribution.
    - **Interpretation:** Lower is better. 
        - *Good:* Trending towards 0.
        - *Bad:* Stuck at a high value (e.g., > 2.0 for 5 classes) or increasing.
    
    **2. Accuracy**
    - **What it is:** The percentage of correct predictions.
    - **Interpretation:** Higher is better.
        - *Good:* > 85% for this task.
        - *Bad:* Near random guessing (1/5 = 20%).
        
    **3. Precision**
    - **What it is:** Out of all images predicted as "Happy", how many were actually "Happy"?
    - **Interpretation:** High precision means fewer False Positives.
    
    **4. Recall**
    - **What it is:** Out of all actual "Happy" images, how many did we correctly predict?
    - **Interpretation:** High recall means fewer False Negatives.
    
    **5. F1-Score**
    - **What it is:** Harmonic mean of Precision and Recall.
    - **Interpretation:** A balanced view of performance.
    
    **6. Confusion Matrix**
    - **What it is:** A grid showing actual vs. predicted classes.
    - **Interpretation:** The diagonal should be dark/high numbers. Off-diagonal elements represent errors.
    """
    return text

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
