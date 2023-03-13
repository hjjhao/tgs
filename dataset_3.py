import os
import cv2
import torch
import config
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

def get_list():
    train_list = os.listdir(os.path.join(config.TRAIN_DATA_PATH, 'images'))
    test_list = os.listdir(os.path.join(config.TEST_DATA_PATH, 'images'))
    train_list, valid_list = train_test_split(train_list, test_size=config.TEST_SPLIT, random_state=config.RANDOM_STATE)
    return train_list, valid_list, test_list

def get_depth():
    depth_csv = pd.read_csv(config.DEPTH_DATA_PATH, index_col='id')
    train_csv = pd.read_csv(os.path.join(config.DATA_ROOT, 'train.csv'), index_col='id')
    train_depth_csv = depth_csv[depth_csv.index.isin(train_csv.index)]
    test_depth_csv = depth_csv[~depth_csv.index.isin(train_csv.index)]
    return train_depth_csv, test_depth_csv

class SaltDataset(Dataset):
    def __init__(self, list, depth_csv, mode, transform=None):
        super(SaltDataset, self).__init__()
        self.list = list
        self.mode = mode
        self.depth_csv = depth_csv
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        filename = self.list[index].split('.')[0]


        if self.mode == 'train' or self.mode == 'valid':
            image = cv2.imread(os.path.join(config.TRAIN_DATA_PATH, 'images', self.list[index]))
            mask = cv2.imread(os.path.join(config.TRAIN_DATA_PATH, 'masks', self.list[index]))
            depth = torch.tensor(self.depth_csv[self.depth_csv.index == filename]['z'])
        elif self.mode == 'test':
            image = cv2.imread(os.path.join(config.TEST_DATA_PATH, 'images', self.list[index]))
            depth = torch.tensor(self.depth_csv[self.depth_csv.index == filename]['z'])

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                image = self.transform(image)
            
            return self.list[index], image, depth
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask, depth



if __name__ == '__main__':
    # train_list, valid_list, test_list = get_list()
    img = cv2.imread('./datasets/train/images/0a0814464f.png')
    print(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    img = transforms.ToPILImage()(np.uint8(img))
    img = transforms.ToTensor()(img)
    print(img.size())