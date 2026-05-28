"""Evaluation script for fault diagnosis models"""

import torch
from config import get_args
from models import SMAConvformer
from utils.train_val_test import train_val_test


def evaluate():
    """Main evaluation function"""
    args = get_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    model = SMAConvformer()
    model = model.to(device)
    model.eval()
    
    print('Evaluation started...')
    
    # Run evaluation using train_val_test framework
    operation = train_val_test(args)
    operation.setup(0)
    acc, j = operation.test(0)
    
    print(f'Evaluation completed!')
    print(f'Accuracy: {acc:.4f}, J-score: {j:.4f}')
    

if __name__ == '__main__':
    evaluate()
