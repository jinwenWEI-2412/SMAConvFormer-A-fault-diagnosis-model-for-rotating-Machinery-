#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import numpy as np
import logging

from utils.train_val_test import train_val_test
from datasets.XJTU_gearbox import dataset_save as XJTU_gearbox
from datasets.XJTU_spurgear import dataset_save as XJTU_spurgear
from datasets.OU_bearing import dataset_save as OU_bearing
from datasets.BJTU_rao import dataset_save as BJTU_rao

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='SMAConvformer', help='the name of the model', choices=[
        'Liconvformer', 'CLFormer', 'convoformer_v1_small', 'mcswint','SMAConvformer','EWSNet',
         'MobileNet', 'MobileNetV2', 'ResNet18', 'MSResNet'])
    parser.add_argument('--save_dataset', type=bool, default=False, help='whether saving the dataset')
    parser.add_argument('--normalize_type', type=str, default='0-1'  , help='data normalization methods',
                        choices=['0-1', '-1-1', 'mean-std'])
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for each batch')

    # dataset parameters
    parser.add_argument('--dataset_name', type=str, default='BJTU_rao', help='the name of the dataset',
                        choices=['XJTU_gearbox', 'XJTU_spurgear', 'OU_bearing', 'BJTU_rao'])
    parser.add_argument('--sigma', type=int, default=0.0, help='the level of noise under noise task')

    # optimization information
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='the para of lr scheduler')
    parser.add_argument('--min_lr', type=int, default=1e-6, help='the para of lr scheduler')
    parser.add_argument('--epoch', type=int, default=100, help='the max number of epoch')
    # 学习率调度器参数
    parser.add_argument('--T_0', type=int, default=5, help='CosineAnnealing初始周期')
    parser.add_argument('--T_mult', type=int, default=2, help='CosineAnnealing周期倍增系数')


    # saving results
    parser.add_argument('--operation_num', type=int, default=5,
                        help='the repeat operation of model. If XJTU_spurgear, set 10; otherwise, set 5')
    parser.add_argument('--only_test', type=bool, default=False, help='loading the trained model if only test')

    args = parser.parse_args()
    return args


def get_args():
    """Get parsed arguments"""
    return parse_args()
