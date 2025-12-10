"""
Parameter Counter
-----------------
Utility script to calculate and display the number of parameters
in the model configured in 'configs/config.yaml'.

Usage:
    python count_params.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import EmotionModel


def main():
    """Main function to calculate and display model parameters."""
    config_path = 'configs/config.yaml'
    
    print("=" * 40)
    print("       MODEL PARAMETER COUNTER")
    print("=" * 40)
    
    model = EmotionModel(config_path)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    
    print(f"\nTotal Parameters:      {total_params:>15,}")
    print(f"Trainable Parameters:  {trainable_params:>15,}")
    print(f"Non-Trainable:         {non_trainable:>15,}")
    print(f"\nModel Size (float32):  {total_params * 4 / (1024**2):>12.2f} MB")
    print("=" * 40)


if __name__ == "__main__":
    main()

