"""

experiments/multitask/datamodules.py


Pytorch DataLoader extensions for multitask experiments.

Created by: Friday September 3rd, 2021
Author: Jacob A Rose


"""

import pytorch_lightning as pl
from torchvision import transforms
import torch

from lightning_hydra_classifiers.experiments.transfer_experiment import TransferExperiment

__all__ = ["MultiTaskDataModule"]




class MultiTaskDataModule(pl.LightningDataModule):
#     valid_tasks = (0, 1)
    
    def __init__(self, 
                 batch_size,
                 task_id: int=0,
                 image_size: int=224,
                 image_buffer_size: int=32,
                 num_workers: int=4,
                 pin_memory: bool=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        
        self.experiment = TransferExperiment()
        self.set_task(task_id)        
        
        self.image_size = image_size
        self.image_buffer_size = image_buffer_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Train augmentation policy
        self.__init_transforms()
        self.tasks = self.experiment.get_multitask_datasets(train_transform=self.train_transform,
                                                            val_transform=self.val_transform)

    def __init_transforms(self):
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.image_size,
                                         scale=(0.25, 1.2),
                                         ratio=(0.7, 1.3),
                                         interpolation=2),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(self.mean, self.std),
            transforms.Grayscale(num_output_channels=3)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size+self.image_buffer_size),
            transforms.ToTensor(),
            transforms.CenterCrop(self.image_size),
            transforms.Normalize(self.mean, self.std),
            transforms.Grayscale(num_output_channels=3)            
        ])

    def set_task(self, task_id: int):
        assert task_id in self.experiment.valid_tasks
        self.task_id = task_id
        

    @property
    def current_task(self):
        return self.tasks[self.task_id]

    def setup(self, stage=None):
        task = self.current_task
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = task['train']
            self.val_dataset = task['val']
            
            self.classes = self.train_dataset.classes
            self.num_classes = len(self.train_dataset.label_encoder)
            self.label_encoder = self.train_dataset.label_encoder
            
        elif stage == 'test':
            self.test_dataset = task['test']
                        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)
