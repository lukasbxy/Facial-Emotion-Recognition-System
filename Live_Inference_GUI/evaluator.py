"""
Evaluator Module
----------------
Manages batch evaluation of the model against a holdout dataset (folder structure).
Calculates and saves metrics including Accuracy, Precision, Recall, and F1-Score.
"""

import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image
import cv2

class Evaluator:
    def __init__(self, model_wrapper, data_dir, history_file="evaluation_history.json"):
        self.model_wrapper = model_wrapper
        self.data_dir = data_dir
        self.history_file = history_file
        if os.path.exists(data_dir):
            self.classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
        else:
            self.classes = []
        
    def get_all_images(self):
        """Returns a list of (image_path, ground_truth_label) tuples."""
        image_list = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_list.append((os.path.join(class_dir, fname), class_name))
        return image_list

    def evaluate(self, progress_callback=None):
        """
        Runs evaluation on all images.
        progress_callback(current, total, current_image_path, predicted_label, true_label)
        """
        images = self.get_all_images()
        total = len(images)
        
        y_true = []
        y_pred = []
        
        results = {
            "timestamp": time.time(),
            "total_images": total,
            "correct_count": 0,
            "accuracy": 0.0,
            "per_class": {}
        }
        
        for i, (img_path, true_label) in enumerate(images):
            # Load Image
            # ModelWrapper expects BGR numpy array (OpenCV format)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Predict
            # We assume the image IS the face, so we don't need face detection here?
            # The user said "test_holdout bilder", usually these are cropped faces if it's an emotion dataset.
            # If they are full scenes, we might need detection. 
            # Given the folder structure (Angry, Happy...), it looks like a classification dataset (cropped).
            # We will pass it directly to predict.
            
            try:
                pred_label, conf, _ = self.model_wrapper.predict(img)
            except Exception as e:
                print(f"Error predicting {img_path}: {e}")
                pred_label = "Error"
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            if pred_label == true_label:
                results["correct_count"] += 1
            
            if progress_callback:
                progress_callback(i + 1, total, img_path, pred_label, true_label)
                
        # Calculate Metrics
        results["accuracy"] = accuracy_score(y_true, y_pred)
        
        # Detailed Report
        report = classification_report(y_true, y_pred, target_names=self.classes, output_dict=True, zero_division=0)
        
        for cls in self.classes:
            if cls in report:
                results["per_class"][cls] = {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1-score": report[cls]["f1-score"],
                    "support": report[cls]["support"]
                }
        
        self.save_results(results)
        return results

    def save_results(self, results):
        # Load existing history
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = [history]
            except:
                history = []
        
        # Append new result
        history.append(results)
        
        # Save
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=4)
            
    def get_last_comparison(self):
        """Returns (current_result, previous_result) or None."""
        if not os.path.exists(self.history_file):
            return None, None
            
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
                
            if not history:
                return None, None
                
            current = history[-1]
            previous = history[-2] if len(history) > 1 else None
            return current, previous
        except:
            return None, None
