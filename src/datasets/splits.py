import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

from .transforms import provide_image_transforms


class ImageSubsetTransformed(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.img_paths = dataset.img_paths[indices]
        self.labels = dataset.labels[indices]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TextSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.labels = dataset.labels[indices]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def disjoint_train_test_idx(dataset, test_size=0.5, random_state=0):
    if test_size == 1.0:
        return np.array([], dtype=int), np.arange(len(dataset))
    splitter = GroupShuffleSplit(
        n_splits=2,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(None, None, dataset.labels))
    return train_idx, test_idx


def image_dataset_split_transform(dataset, test_size=0.5, random_state=0):
    if random_state == 0 and hasattr(dataset, 'train_idx'):
        train_idx, test_idx = dataset.train_idx, dataset.test_idx
    train_idx, test_idx = disjoint_train_test_idx(dataset, test_size, random_state)
    train_transforms, test_transforms = provide_image_transforms()
    train_dataset = ImageSubsetTransformed(dataset, train_idx, transform=train_transforms)
    test_dataset = ImageSubsetTransformed(dataset, test_idx, transform=test_transforms)
    assert set(train_dataset.labels).isdisjoint(set(test_dataset.labels))
    return train_dataset, test_dataset


def text_dataset_split_transform(dataset, test_size=0.5, random_state=0):
    train_idx, test_idx = disjoint_train_test_idx(dataset, test_size, random_state)
    train_dataset = TextSubset(dataset, train_idx)
    test_dataset = TextSubset(dataset, test_idx)
    assert set(train_dataset.labels).isdisjoint(set(test_dataset.labels))
    return train_dataset, test_dataset
