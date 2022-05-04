import numpy as np
import pandas as pd
import scipy.io as sio

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class Cars196Dataset(Dataset):
    def __init__(self, root='cars196', download=False):
        self.root = root
        self.load_meta()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return img, label
    
    def load_meta(self):
        mat = sio.loadmat(f'{self.root}/cars_annos.mat', squeeze_me=True)
        self.labels = mat['annotations']['class'].astype(int) - 1
        self.img_paths = np.array([Path(self.root) / fname for fname in mat['annotations']['relative_im_path']])


class SOPDataset(Dataset):
    def __init__(self, root='sop', download=False):
        self.root = root
        self.load_meta()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return img, label

    def load_meta(self):
        interfix = 'Stanford_Online_Products'
        df_train = pd.read_csv(f'{self.root}/{interfix}/Ebay_train.txt', sep=' ')
        df_test = pd.read_csv(f'{self.root}/{interfix}/Ebay_test.txt', sep=' ')
        self.labels = np.hstack([df_train.class_id.to_numpy(), df_test.class_id.to_numpy()])
        self.train_idx = np.arange(len(df_train))
        self.test_idx = np.arange(len(df_test)) + len(df_train)
        paths = np.hstack([df_train.path.to_numpy(), df_test.path.to_numpy()])
        self.img_paths = np.array([Path(self.root) / interfix / fname for fname in paths])


class CUB200Dataset(Dataset):
    def __init__(self, root='cub', download=False):
        self.root = root
        self.load_meta()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return img, label

    def load_meta(self):
        interfix = 'CUB_200_2011'
        samples = ImageFolder(f'{self.root}/{interfix}/images').samples
        self.img_paths = np.array([s[0] for s in samples])
        self.labels = np.array([s[1] for s in samples])


class Dogs130Dataset(Dataset):
    def __init__(self, root='dogs', download=False):
        self.root = root
        self.load_meta()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return img, label

    def load_meta(self):
        interfix = 'low-resolution'
        samples = ImageFolder(f'{self.root}/{interfix}').samples
        self.img_paths = np.array([s[0] for s in samples])
        self.labels = np.array([s[1] for s in samples])
