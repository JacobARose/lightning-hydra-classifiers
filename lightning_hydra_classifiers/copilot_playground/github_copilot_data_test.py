
# from omegaconf import DictConfig
# import pytorch_lightning as pl
# from torchvision import transforms
# from typing import *





import pandas as pd


def intersection(data_df: pd.DataFrame, other_df: pd.DataFrame, id_col: str="catalog_number", suffixes=("_x", "_y")) -> pd.DataFrame:
    """
    Return a new dataframe containing only rows that share the same values for `id_col` between `data_df` and `other_df`
    
    Equivalent to an AND join between sets
    """

    return data_df.merge(other_df, how='inner', on=id_col, suffixes=suffixes)


def left_exclusive(data_df: pd.DataFrame, other_df: pd.DataFrame, id_col: str="catalog_number") -> pd.DataFrame:
    """
    Return a new dataframe containing only rows from `data_df` that do not share an `id_col` value with any row from `other_df`.
    
    Equivalent to subtracting the set of `id_col` values in `other_df` from `data_df`
    """
    
    omit = list(other_df[id_col].values)
    
    return data_df[data_df[id_col].apply(lambda x: x not in omit)]














# class LeavesLightningDataModule(pl.LightningDataModule): #pl.LightningDataModule):
        
#     def __init__(self,
#                  config: DictConfig,
#                  data_dir: Optional[str]=None):
#         """
#         Custom Pytorch lightning datamodule for accessing train, validation, and test data splits.
#         """


#     def train_dataloader(self):
#         return self.train_loader



#     def show_batch(self, batch):
#         """
#         Display a batch of images
#         """