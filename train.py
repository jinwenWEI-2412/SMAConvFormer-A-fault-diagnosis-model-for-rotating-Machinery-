"""Main training script for fault diagnosis models"""

import torch
import argparse
import os
from datetime import datetime
from config import get_args
from models import SMAConvformer
from utils import preprocess_data
from utils.train_val_test import train_val_test
from utils.logger import setlogger
from datasets import BJTU_rao, XJTU_gearbox, XJTU_spurgear, OU_bearing
import numpy as np
import logging


def main():
    """Main training function"""
    args = get_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create result directory
    save_dir = os.path.join('./results/{}'.format(args.dataset_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set up logging
    setlogger(os.path.join(save_dir, args.model_name + '.log'))
    
    # Save arguments
    logging.info("\n")
    time = datetime.strftime(datetime.now(), '%m-%d %H:%M:%S')
    logging.info('{}'.format(time))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    
    # Handle dataset saving if requested
    if args.save_dataset:
        if args.dataset_name == 'XJTU_gearbox':
            XJTU_gearbox(args)
        elif args.dataset_name == 'XJTU_spurgear':
            XJTU_spurgear(args)
        elif args.dataset_name == 'OU_bearing':
            OU_bearing(args)
        elif args.dataset_name == 'BJTU_rao':
            BJTU_rao(args)
    else:
        # Initialize model and start training
        Accuracy = []
        J = []
        operation = train_val_test(args)
        
        for i in range(args.operation_num):
            if args.only_test == 0:
                operation.setup(i)
                operation.train_val(i)
            else:
                operation.setup(i)
            acc, j = operation.test(i)
            Accuracy.append(acc)
            J.append(j)
            
            if i == 4 or i == 9:
                Accuracy_arr = np.array(Accuracy) * 100
                Accuracy_mean = Accuracy_arr.mean()
                Accuracy_var = Accuracy_arr.var()
                Accuracy_max = Accuracy_arr.max()
                Accuracy_min = Accuracy_arr.min()
                J_arr = np.array(J)
                J_mean = J_arr.mean()
                J_var = J_arr.var()
                J_max = J_arr.max()
                J_min = J_arr.min()
                
                Accuracy_list = ', '.join(['{:.2f}'.format(acc) for acc in Accuracy_arr])
                J_list = ', '.join(['{:.2f}'.format(j) for j in J_arr])
                logging.info('All acc: {}, \nMean acc: {:.2f}, Var acc {:.2f}, Max acc {:.2f}, Min acc {:.2f}'.format(
                    Accuracy_list, Accuracy_mean, Accuracy_var, Accuracy_max, Accuracy_min))
                logging.info('All J: {}, \nMean J: {:.2f}, Var J {:.2f}, Max J {:.2f}, Min J {:.2f}\n'.format(
                    J_list, J_mean, J_var, J_max, J_min))
                Accuracy = []
                J = []


if __name__ == '__main__':
    main()
