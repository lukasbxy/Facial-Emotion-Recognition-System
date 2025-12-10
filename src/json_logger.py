"""
JSON Logger
-----------
Handles logging of training metrics to a JSON file for easy consumption by the dashboard.
Supports:
- Epoch and Batch level logging
- Run archival
- System info logging
"""

import json
import os
import time

class JSONLogger:
    def __init__(self, log_dir, resume=False):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'metrics.json')
        self.index_file = os.path.join(log_dir, 'runs_index.json')
        
        if not resume and os.path.exists(self.log_file):
            self._archive_old_run()
            self._init_new_data()
        elif os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.data = json.load(f)
                print(f"Resumed logging from {self.log_file}")
            except json.JSONDecodeError:
                print("Warning: Could not decode existing metrics.json. Starting fresh.")
                self._init_new_data()
        else:
            self._init_new_data()
            
        self._update_index()
        self._write_to_disk()

    def _archive_old_run(self):
        # Generate timestamp for archive
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_name = f"metrics_{timestamp}.json"
        archive_path = os.path.join(self.log_dir, archive_name)
        
        # Rename current file
        try:
            os.rename(self.log_file, archive_path)
            print(f"Archived previous run to {archive_name}")
            
            # Update index with archived run
            self._add_to_index(archive_name, timestamp)
        except OSError as e:
            print(f"Error archiving run: {e}")

    def _add_to_index(self, filename, timestamp):
        index_data = []
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
            except:
                pass
        
        # Add new run to start of list
        index_data.insert(0, {
            "id": filename,
            "name": f"Run {timestamp}",
            "timestamp": timestamp,
            "filename": filename
        })
        
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f, indent=2)

    def _update_index(self):
        # Ensure runs_index.json exists and has the current run if needed
        # Actually, the frontend can just assume "Current Run" is metrics.json
        # But providing a consistent list is better.
        # Let's just make sure the file exists so frontend doesn't 404
        if not os.path.exists(self.index_file):
            with open(self.index_file, 'w') as f:
                json.dump([], f)

    def _init_new_data(self):
        self.data = {
            "status": "Starting...",
            "epoch": 0,
            "total_epochs": 0,
            "system_info": {},
            "history": [],
            "educational_notes": {
                "loss": "Loss measures error. Lower is better. If Train Loss goes down but Val Loss goes up, you are overfitting.",
                "accuracy": "Percentage of correct guesses. Higher is better. >85% is great for this task.",
                "f1": "Balance between Precision and Recall. Good for uneven datasets."
            }
        }

    def log_system_info(self, info):
        self.data["system_info"] = info
        self._write_to_disk()

    def set_total_epochs(self, epochs):
        self.data["total_epochs"] = epochs
        self._write_to_disk()

    def log_epoch(self, epoch, metrics):
        """
        metrics: dict with keys like 'train_loss', 'val_loss', 'val_acc', etc.
        """
        self.data["epoch"] = epoch
        self.data["status"] = "Training"
        
        # Add timestamp
        metrics["timestamp"] = time.time()
        metrics["epoch"] = epoch
        
        self.data["history"].append(metrics)
        self._write_to_disk()

    def log_batch(self, epoch, batch_loss, batch_idx, total_batches):
        """
        Updates the current status with batch-level info without adding to history yet.
        """
        self.data["status"] = f"Training Epoch {epoch+1}/{self.data['total_epochs']} - Batch {batch_idx}/{total_batches} - Loss: {batch_loss:.4f}"
        self.data["current_batch_loss"] = batch_loss
        self._write_to_disk()

    def finish(self):
        self.data["status"] = "Complete"
        self._write_to_disk()

    def _write_to_disk(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
