"""
Emotion Recognition Training Script
-----------------------------------
This script handles the training pipeline for the Emotion Recognition model.
It includes:
- Data loading and augmentation
- Model initialization (ResNet, ConvNeXt, ImprovedCNN)
- Training loop with mixed precision (AMP)
- Validation and metric calculation
- Logging to TensorBoard and JSON
- Automatic recovery from checkpoints

Usage:
    python src/train.py [--config configs/config.yaml]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np
import subprocess
import webbrowser
import time
import sys
import atexit
import threading
import http.server
import socketserver

from dataset import EmotionDataModule
from model import EmotionModel
from utils import calculate_metrics, plot_confusion_matrix, plot_to_image
from utils import SoftTargetCrossEntropy, mixup_data, mixup_criterion
from json_logger import JSONLogger
from tuner import find_optimal_batch_size, find_optimal_num_workers
from torchvision import datasets, transforms

def start_dashboard_server(directory, port=8000):
    class SilentHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        
        def log_message(self, format, *args):
            pass # Silence server logs

    class SilentTCPServer(socketserver.TCPServer):
        def handle_error(self, request, client_address):
            # Suppress socket errors like connection aborted
            pass

    try:
        with SilentTCPServer(("", port), SilentHandler) as httpd:
            print(f"Dashboard serving at port {port}")
            httpd.serve_forever()
    except Exception as e:
        print(f"Dashboard server error: {e}")

def get_device_choice():
    print("\n" + "="*40)
    print("       DEVICE SELECTION")
    print("="*40)
    print("1. CPU")
    
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except:
            gpu_name = "Unknown GPU"
        print(f"2. GPU [{gpu_name}]")
    else:
        print("2. GPU [Not detected/installed]")
    
    print("="*40)
    
    while True:
        choice = input("Choose device (enter 1 or 2): ").strip()
        if choice == '1':
            return 'cpu'
        elif choice == '2':
            if gpu_available:
                return 'cuda'
            else:
                print("\n>>> WARNING: GPU selected but not available. Falling back to CPU.")
                print(">>> Tip: Did you install the 'cpuonly' version of PyTorch?")
                print(">>> To fix, run: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia\n")
                time.sleep(2)
                return 'cpu'
        print("Invalid choice. Please enter 1 or 2.")

def train(config_path, auto_tune=False, device_arg=None):
    # Setup Device Interactive
    if device_arg:
        device_name = device_arg
    else:
        device_name = get_device_choice()
    
    device = torch.device(device_name)
    print(f"\n>>> Starting training on: {device}\n")

    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- AUTO-TUNING ---
    if auto_tune and device.type == 'cuda':
        print("\n>>> Starting Auto-Tuning Process...")
        
        # We need a temporary dataset instance for tuning
        data_dir = config['data']['data_dir']
        image_size = config['data']['image_size']
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        temp_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        
        # 1. Tune Batch Size
        best_bs = find_optimal_batch_size(
            lambda: EmotionModel(config_path), 
            temp_dataset, 
            device
        )
        config['data']['batch_size'] = best_bs
        
        # 2. Tune Num Workers
        best_workers = find_optimal_num_workers(temp_dataset, best_bs)
        config['data']['num_workers'] = best_workers
        
        print("\n" + "="*40)
        print(f"   TUNING COMPLETE")
        print(f"   Batch Size: {best_bs}")
        print(f"   Num Workers: {best_workers}")
        print("="*40 + "\n")
        
        # Clean up
        del temp_dataset
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # Setup Data (with potentially updated config)
    dm = EmotionDataModule(config) # Pass dict, not path
    train_loader, val_loader, class_names = dm.get_dataloaders()
    
    # --- Checkpoint / Resume Logic (Pre-check) ---
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    resume_training = os.path.exists(checkpoint_path)

    # Setup Logging
    json_logger = JSONLogger(config['training']['log_dir'], resume=resume_training)
    json_logger.set_total_epochs(config['training']['epochs'])

    # Log System Info
    json_logger.log_system_info({
        "batch_size": config['data']['batch_size'],
        "num_workers": config['data']['num_workers'],
        "device": str(device),
        "image_size": config['data']['image_size'],
        "model": config['model']['name']
    })
    
    # Setup Model
    model = EmotionModel(config_path).to(device)
    
    # Setup Optimization
    # Check for augmentation config
    aug_config = config.get('augmentation', {})
    mixup_alpha = aug_config.get('mixup_alpha', 0.0)
    cutmix_alpha = aug_config.get('cutmix_alpha', 0.0)
    use_mixup = mixup_alpha > 0 or cutmix_alpha > 0
    
    label_smoothing = aug_config.get('label_smoothing', 0.0)
    
    if use_mixup:
        print(f">>> Mixup/CutMix Enabled (Alpha: {mixup_alpha}/{cutmix_alpha})")
        # We use mixup_criterion which mixes losses, so we need a criterion that accepts integer labels
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # AdamW with specific settings
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Scheduler with Warmup
    epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    
    if warmup_epochs > 0:
        # Sequential LR: Warmup -> Cosine
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=warmup_epochs
        )
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    
    # Setup Scaler for AMP
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # --- Checkpoint / Resume Logic ---
    start_epoch = 0

    if resume_training:
        print(f"\n>>> Found checkpoint at {checkpoint_path}")
        print(">>> Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f">>> Resuming from Epoch {start_epoch+1}\n")
        except RuntimeError as e:
            print(f"\n>>> ERROR: Checkpoint architecture mismatch!")
            print(f">>> The checkpoint at {checkpoint_path} seems to be from a different model architecture.")
            print(f">>> Error details: {str(e)[:100]}...")
            
            archive_name = f"last_checkpoint_archived_{int(time.time())}.pth"
            archive_path = os.path.join(checkpoint_dir, archive_name)
            os.rename(checkpoint_path, archive_path)
            
            print(f">>> Archived incompatible checkpoint to: {archive_name}")
            print(">>> Starting training from scratch...\n")
            start_epoch = 0
    else:
        print("\n>>> No checkpoint found. Starting from scratch.\n")

    # --- Launch Custom Dashboard ---
    print("Launching Custom Dashboard...")
    dashboard_thread = threading.Thread(target=start_dashboard_server, args=(os.getcwd(), 8000))
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    time.sleep(1)
    print("Opening Dashboard in Browser...")
    webbrowser.open("http://localhost:8000/dashboard/index.html")
    # --------------------------
    
    # Create Checkpoint Dir
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    try:
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            print(f"Epoch {epoch+1}/{epochs}")
            
            # --- TRAINING ---
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (images, labels) in enumerate(loop):
                images, labels = images.to(device), labels.to(device)
                
                # Mixup/CutMix
                if use_mixup:
                    images, targets_a, targets_b, lam = mixup_data(
                        images, labels, mixup_alpha, device=device.type
                    )
                
                # Forward with AMP
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    if use_mixup:
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = criterion(outputs, labels)
                
                # Backward with AMP
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                
                # Accuracy calculation is tricky with mixup, we just use the argmax for rough tracking
                if use_mixup:
                    # For tracking, we compare against the "dominant" label (lam > 0.5 ? a : b)
                    # But simpler is just to skip strict accuracy or use weighted.
                    # Let's just track it against the original labels if possible, but we don't have them easily if we overwrote.
                    # We have targets_a and targets_b.
                    correct_a = (predicted == targets_a).sum().item()
                    correct_b = (predicted == targets_b).sum().item()
                    train_correct += lam * correct_a + (1 - lam) * correct_b
                else:
                    train_correct += (predicted == labels).sum().item()
                
                # Update Progress Bar
                loop.set_description(f"Train - Loss: {loss.item():.4f}")
                
                # Log Batch Metrics (Throttle to reduce IO)
                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                    json_logger.log_batch(epoch, loss.item(), batch_idx, len(train_loader))

            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # --- VALIDATION ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            # Validation should NOT use Mixup
            val_criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = val_criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate Advanced Metrics
            all_preds = torch.tensor(all_preds)
            all_labels = torch.tensor(all_labels)
            precision, recall, f1 = calculate_metrics(all_preds, all_labels)
            
            # Performance Metrics
            epoch_duration = time.time() - epoch_start_time
            total_samples = len(train_loader.dataset) + len(val_loader.dataset)
            throughput = total_samples / epoch_duration
            
            # Log Epoch Metrics to JSON
            json_logger.log_epoch(epoch, {
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch_duration": epoch_duration,
                "throughput": throughput
            })
            
            # Step Scheduler
            scheduler.step()
            
            print(f"Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")
            
            # Save Checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), os.path.join(config['training']['checkpoint_dir'], f"model_epoch_{epoch+1}.pth"))

            # Save "Latest" Checkpoint for Resuming
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

    except KeyboardInterrupt:
        print("\n\n" + "!"*50)
        print("   TRAINING PAUSED BY USER (Ctrl+C)")
        print("!"*50)
        print(">>> Saving checkpoint for resume...")
        
        save_epoch = epoch - 1 if epoch > 0 else 0
        
        checkpoint = {
            'epoch': save_epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f">>> Progress saved to {checkpoint_path}")
        print(">>> Run the script again to resume from the start of this epoch.")
        sys.exit(0)

    # Save Final Model
    torch.save(model.state_dict(), os.path.join(config['training']['checkpoint_dir'], "model_final.pth"))
    json_logger.finish()
    print("Training Complete!")
    
    print("The Dashboard is still active.")
    try:
        input("Press Enter to close the dashboard and exit...")
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--auto-tune', action='store_true', help='Enable auto-tuning of batch size and workers')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu or cuda)')
    args = parser.parse_args()
    
    train(args.config, args.auto_tune, args.device)
