"""

tests/helpers.py

Useful mock data and model fixtures for use in test suite.


Created by: Jacob A Rose
Creayed on: Friday Oct 22nd, 2021


"""


import torch


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=2000, shape=(3,64,64)):
        self.num_samples = num_samples
        self.shape = shape
        self.data = torch.randn(num_samples, *shape)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

class RandomTupleSupervisedDataset(RandomDataset):
    
    def __init__(self, num_classes=1000, num_samples=2000, shape=(3,64,64)):
        super().__init__(num_samples, shape)
        self.num_classes = num_classes
        
        self.targets = torch.randperm(num_classes)[:num_samples]
        
    def __getitem__(self, index):
        return self.data[index], self.targets[index]