"""
Hyperparameter Auto-Tuning
--------------------------
Provides utilities for automatically determining optimal training parameters:
- Batch Size: Finds the maximum batch size that fits in GPU memory.
- Num Workers: Finds the optimal number of data loading workers for throughput.
"""

import torch
import time
import copy
from torch.utils.data import DataLoader
import gc

def find_optimal_batch_size(model_class, dataset, device, start_batch_size=32, max_batch_size=8192):
    print("\n" + "="*40)
    print("   AUTO-TUNING: BATCH SIZE")
    print("="*40)
    
    batch_size = start_batch_size
    best_batch_size = start_batch_size
    
    # Create a dummy model for testing memory
    # model_class is expected to be a factory function returning a model instance
    model = model_class().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Testing batch sizes on {device}...")
    
    try:
        while batch_size <= max_batch_size:
            print(f"Testing batch size: {batch_size}...", end="", flush=True)
            
            try:
                # Create a temporary loader
                # We use a subset to be fast
                subset_indices = range(min(len(dataset), batch_size * 5)) # Enough for a few steps
                subset = torch.utils.data.Subset(dataset, subset_indices)
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
                
                # Run a few steps
                start_time = time.time()
                steps = 0
                for images, labels in loader:
                    if steps >= 3: break
                    
                    images, labels = images.to(device), labels.to(device)
                    
                    # Full training step simulation
                    with torch.amp.autocast('cuda', enabled=True):
                        outputs = model(images)
                        loss = torch.nn.functional.cross_entropy(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    steps += 1
                
                # If we got here, it worked
                print(" OK")
                best_batch_size = batch_size
                batch_size *= 2
                
                # Cleanup
                del loader
                gc.collect()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(" OOM (Out of Memory)!")
                    torch.cuda.empty_cache()
                    break
                else:
                    print(f" Error: {e}")
                    break
    except Exception as e:
        print(f"Unexpected error during tuning: {e}")
    finally:
        del model
        del optimizer
        del scaler
        torch.cuda.empty_cache()
    
    # Safe fallback: Use 80-90% of the max successful batch size to be safe during long runs
    # But since we doubled, 'best_batch_size' is the last one that worked.
    # Let's stick with that.
    print(f">>> Optimal Batch Size Found: {best_batch_size}")
    return best_batch_size

def find_optimal_num_workers(dataset, batch_size, max_workers=16):
    print("\n" + "="*40)
    print("   AUTO-TUNING: NUM WORKERS")
    print("="*40)
    
    import os
    max_workers = min(max_workers, os.cpu_count())
    best_workers = 0
    best_time = float('inf')
    
    print(f"Testing num_workers (0 to {max_workers})...")
    
    # Use a reasonable subset
    subset_indices = range(min(len(dataset), batch_size * 10))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    for workers in range(0, max_workers + 1, 2): # Step by 2 to save time
        if workers == 0: workers = 0 # Explicit check
        
        print(f"Testing workers: {workers}...", end="", flush=True)
        
        loader = DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=workers,
            persistent_workers=(workers > 0),
            pin_memory=True
        )
        
        start_time = time.time()
        for _ in loader:
            pass
        end_time = time.time()
        duration = end_time - start_time
        
        print(f" Time: {duration:.4f}s")
        
        if duration < best_time:
            best_time = duration
            best_workers = workers
            
    print(f">>> Optimal Num Workers Found: {best_workers}")
    return best_workers
