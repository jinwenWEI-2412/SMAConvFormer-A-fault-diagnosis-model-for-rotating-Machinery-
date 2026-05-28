"""Main training script for fault diagnosis models"""

import torch
import argparse
from config import get_args
from models import SMAConvformer
from datasets import preprocess_data


def main():
    """Main training function"""
    args = get_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    model = SMAConvformer()
    model = model.to(device)
    
    # Load and preprocess data
    # train_loader, val_loader, test_loader = preprocess_data(args)
    
    print('Training started...')
    # Add training loop here
    

if __name__ == '__main__':
    main()
