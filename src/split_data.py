"""
Data Split Utility
------------------
Utility script to create a holdout test set by moving a percentage
of images from the training data to a separate test directory.

Usage:
    python src/split_data.py
"""

import os
import shutil
import random
import math


def split_test_data(source_dir, target_dir, split_percentage=0.05):
    """
    Moves a percentage of files from source_dir to target_dir for each class.
    """
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # Get list of classes (subdirectories)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found classes: {classes}")
    print(f"Moving {split_percentage*100}% of data to {target_dir}...")

    for class_name in classes:
        class_source_path = os.path.join(source_dir, class_name)
        class_target_path = os.path.join(target_dir, class_name)
        
        # Create target class directory
        os.makedirs(class_target_path, exist_ok=True)
        
        # Get all files
        files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
        total_files = len(files)
        num_to_move = math.ceil(total_files * split_percentage)
        
        # Randomly select files
        files_to_move = random.sample(files, num_to_move)
        
        print(f"  {class_name}: Moving {num_to_move}/{total_files} files.")
        
        for file_name in files_to_move:
            src = os.path.join(class_source_path, file_name)
            dst = os.path.join(class_target_path, file_name)
            shutil.move(src, dst)

    print("Done!")

if __name__ == "__main__":
    # Define paths
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "archive", "data")
    test_dir = os.path.join(base_dir, "archive", "test_holdout") # New folder for holdout set
    
    split_test_data(data_dir, test_dir, split_percentage=0.05)
