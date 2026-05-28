import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import pickle

from datasets.data_pre import *

label = [i for i in range(0, 7)]

# load dataset
class dataset(Dataset):
    def __init__(self, list_data, transform):
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()
        self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        lab = self.labels[item]
        seq = self.transforms(seq)
        return seq, lab

# save dataset
def dataset_save(args):
    data1 = []
    lab1 = []
    data_dir = 'D:\item\SMAConvFormer-A fault diagnosis model for rotating Machinery with high noise and variable operating conditions\data\BJTU_rao'
    fault_files = [
        '1.csv',
        '2.csv',
        '3.csv',
        '4.csv',
        '5.csv',
        '6.csv',
        '7.csv',
        '8.csv',
    ]

    for i in tqdm(range(7)):  # the number of fault mode
        path = os.path.join(data_dir, fault_files[i])
        data, lab = data_load(args, path, label=label[i])
        data1 += data
        lab1 += lab

    # creat the saving file
    if not os.path.exists('./data/save_dataset'):
        os.makedirs('./data/save_dataset')
    list_data = [data1, lab1]
    # 使用 pickle 保存数据
    with open('./data/save_dataset/' + args.dataset_name + '.pkl', 'wb') as f:
        pickle.dump(list_data, f)

# load data from the file
def data_load(args, root, label):
    data = []
    lab = []
    df = pd.read_csv(root)
    ch11 = df['CH11'].values
    ch12 = df['CH12'].values
    ch13 = df['CH13'].values

    length = 64000 * 10  # 64KHZ * 10 seconds
    start, end = 0, 1024
    while end <= length:
        x1 = ch11[start:end]
        x2 = ch12[start:end]
        x3 = ch13[start:end]
        sample = np.stack((x1, x2, x3), axis=0)
        data.append(sample)
        lab.append(label)
        start += 1050
        end += 1050
    return data, lab

class BJTU_rao(object):
    num_sensor = 3
    num_classes = 5

    # load dataset for operation
    def data_prepare(self, args, op_num):
        # 使用 pickle 加载数据
        with open('./data/save_dataset/' + args.dataset_name + '.pkl', 'rb') as f:
            list_data = pickle.load(f)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        # different operations using different datasets to train, val and test
        train_pd, val_test_pd = train_test_split(data_pd, test_size=7/12,
                                                 random_state=op_num, stratify=data_pd["label"])
        val_pd, test_pd = train_test_split(val_test_pd, test_size=4/7,
                                           random_state=op_num, stratify=val_test_pd["label"])
        # test_pd = test_pd.sort_values('label')  # sorting the test set

        # the way of data preprocess
        train_preprocess = Compose([Normalize(args.normalize_type), Retype()])
        test_preprocess = Compose([RandomAddGaussian(args.sigma), RandomScale(args.sigma),
                                   Normalize(args.normalize_type), Retype()])

        train_dataset = dataset(list_data=train_pd, transform=train_preprocess)
        val_dataset = dataset(list_data=val_pd, transform=train_preprocess)
        test_dataset = dataset(list_data=test_pd, transform=test_preprocess)
        return train_dataset, val_dataset, test_dataset