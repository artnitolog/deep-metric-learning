import numpy as np
import pandas as pd
import scipy.io as sio

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class WOS134Dataset(Dataset):
    def __init__(self, tokenizer, root='wos134', download=False, device='cpu'):
        self.root = root
        self.device = device
        self.pretokenize_texts(tokenizer)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.att_masks[idx]), self.labels[idx]
    
    def pretokenize_texts(self, tokenizer):
        df = pd.read_csv(f'{self.root}/wos134.csv')
        self.texts = list(df.text)
        self.labels = df.label.to_numpy()
        t = tokenizer(self.texts, padding=True, truncation=True, return_tensors="pt")
        self.input_ids, self.att_masks = t["input_ids"].to(self.device), t["attention_mask"].to(self.device)


class News20Dataset(Dataset):
    def __init__(self, tokenizer, root='newsgroups20', download=False, device='cpu'):
        self.root = root
        self.device = device
        self.pretokenize_texts(tokenizer)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.att_masks[idx]), self.labels[idx]

    def pretokenize_texts(self, tokenizer):
        df = pd.read_csv(f'{self.root}/newsgroups20.csv')
        self.texts = list(df.text.astype(str))
        self.labels = df.label.to_numpy()
        t = tokenizer(self.texts, padding=True, truncation=True, return_tensors="pt")
        self.input_ids, self.att_masks = t["input_ids"].to(self.device), t["attention_mask"].to(self.device)


