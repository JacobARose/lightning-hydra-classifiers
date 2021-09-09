"""

callbacks/image_stats_callbacks.py


Pytorch Lightning Callbacks for analyzing image statistics

Created on: Wednesday September 8th, 2021
Author: Jacob A Rose


"""



from torch import nn
import torch
from pathlib import Path
import os

from lightning_hydra_classifiers.utils.template_utils import get_logger
logger = get_logger(name=__name__)


class ImageStatsAccumulator(nn.Module):
    """
    Calculates a dataset-wide set of image statistics.
    Currently:
        - per-channel pixel mean
        - per-channel pixel std
        - per-channel pixel min
        - per-channel pixel max
    
    """
    
    def __init__(self, name: str="image_stats", cache_dir: str=None, clear_cache: bool=True):
        self.pixel_count = torch.zeros(1)
        self.image_count = torch.zeros(1)
        self.sum_of_pixels = torch.zeros(3)
        self.sum_of_square_pixels = torch.zeros(3)
        self.resolution = torch.zeros(1)
#         self.channels = torch.Tensor(3)
        
        self.global_max_pixel = torch.Tensor([-float("inf")]*3)
        self.global_min_pixel = torch.Tensor([float("inf")]*3)
        
        self.name = name
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(self.cache_dir, self.name + ".pth")
#         self.clear_cache = clear_cache
        if os.path.isfile(self.cache_path):
            if clear_cache:
                logger.warning(f"Removing prior cache prior to stats update.")
                os.remove(self.cache_path)
            else:
                self.load_from_cache()

        
    def compute(self):        
        global_mean = self.sum_of_pixels / self.pixel_count
        global_var = (self.sum_of_square_pixels / self.pixel_count) - global_mean**2
        global_std = torch.sqrt(global_var)
        return global_mean, global_std, self.global_min_pixel, self.global_max_pixel
    
    
    def update_step(self, batch):
        images = batch[0]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        if self.resolution == 0:
            self.resolution = h
        assert self.resolution == h == w
        
        batch_max = torch.amax(images, dim=(0,2,3))
        batch_min = torch.amin(images, dim=(0,2,3))
        
        self.global_max_pixel = torch.amax(torch.stack([self.global_max_pixel, batch_max]), dim=0)
        self.global_min_pixel = torch.amin(torch.stack([self.global_min_pixel, batch_min]), dim=0)
        
        
        self.sum_of_pixels        += torch.sum(images,    dim=[0, 2, 3])
        self.sum_of_square_pixels += torch.sum(images**2, dim=[0, 2, 3])
        self.pixel_count          += nb_pixels
        self.image_count          += b
        
        
    def update(self, loader: torch.utils.data.DataLoader):

        for batch in tqdm(loader):
            self.update_step(batch)
        global_mean, global_std, global_min, global_max = self.compute()
        logger.info(f"[FINISHED] Calculating full dataset stats from a collection of {self.image_count} images at resolution {self.resolution}")
        logger.info(f"Global channel-wise mean = {global_mean}")
        logger.info(f"Global channel-wise std = {global_std}")
        logger.info(f"Global channel-wise pixel min = {global_min}")
        logger.info(f"Global channel-wise pixel max = {global_max}")

        return global_mean, global_std, global_min, global_max
    
    
    def state_dict(self):
        global_mean, global_std, _, _ = self.compute()
        return {"global_mean": global_mean,
                "global_std": global_std,
                "pixel_count": self.pixel_count,
                "image_count": self.image_count,
                "sum_of_pixels": self.sum_of_pixels,
                "sum_of_square_pixels": self.sum_of_square_pixels,
                "resolution": self.resolution,
                "global_max_pixel": self.global_max_pixel,
                "global_min_pixel": self.global_min_pixel}
    
    def load_state_dict(self, state_dict):
        self.pixel_count = state_dict["pixel_count"]
        self.image_count = state_dict["image_count"]
        self.sum_of_pixels = state_dict["sum_of_pixels"]
        self.sum_of_square_pixels = state_dict["sum_of_square_pixels"]
        self.resolution = state_dict["resolution"] 
        self.global_max_pixel = state_dict["global_max_pixel"]
        self.global_min_pixel = state_dict["global_min_pixel"]
        
        global_mean, global_std, _, _ = self.compute()
        assert torch.all(global_mean == state_dict["global_mean"])
        assert torch.all(global_std == state_dict["global_std"])    
    
    
    def cache(self):
        if not isinstance(self.cache_dir, (str, Path)):
            return
#         self.cache_path = os.path.join(self.cache_dir, self.name + ".pth")
        torch.save(self.state_dict(), self.cache_path)
        logger.info(f"Updated Image Stats cache located at: {self.cache_path}")

    def load_from_cache(self):
        if not isinstance(self.cache_dir, (str, Path)):
            logger.warning(f"No cache detected. No-op.")
            return
#         self.cache_path = os.path.join(self.cache_dir, self.name + ".pth")
        if os.path.isfile(self.cache_path):            
            self.load_state_dict(torch.load(self.cache_path))
            logger.info(f"Loaded Image Stats from cache located at: {self.cache_path}")
        else:
            logger.warning(f"No cache detected. No-op.")
        
    def __repr__(self):
        return "ImageStatsAccumulator:\n" + str('\n'.join([f"{k}: {v}" for k,v in self.state_dict().items()]))

    
    
    
    
    

class InputMonitor(pl.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)