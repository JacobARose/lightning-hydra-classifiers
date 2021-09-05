"""
lightning_hydra_classifiers/experiments/transfer_experiments.py



Created on: Sunday, August 30th, 2021
Author: Jacob A Rose


"""


import argparse
from munch import Munch
import os
from pathlib import Path
from typing import Tuple, Dict

from lightning_hydra_classifiers.data.utils.make_catalogs import CSV_CATALOG_DIR_V1_0, EXPERIMENTAL_DATASETS_DIR
from lightning_hydra_classifiers.data.datasets.common import CSVDatasetConfig, CSVDataset, DataSplitter


__all__ = ["TransferExperiment"]


class TransferExperiment:
    
    valid_tasks = (0, 1)
    
    def __init__(self,
                 config=None):
        self.parse_config(config)
        
    def parse_config(self,
                     config):
        config = config or Munch()
        if "source_root_dir" not in config:
            config.source_root_dir = CSV_CATALOG_DIR_V1_0
        if "experiment_dir" not in config:
            config.experiment_root_dir = EXPERIMENTAL_DATASETS_DIR #"/media/data/jacob/GitHub/lightning-hydra-classifiers/notebooks/experiments_August_2021"
        if "experiment_name" not in config:
            config.experiment_name = "Extant-to-PNAS-512-transfer_benchmark"
            
        self.source_root_dir = config.source_root_dir
        self.experiment_root_dir = config.experiment_root_dir
        self.experiment_name = config.experiment_name
        
        self.experiment_dir = Path(self.experiment_root_dir, self.experiment_name)
        self.config = config
        

#     @staticmethod
    def setup_task_0(self): #experiment_root_dir = Path("/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0/")):
        """
        TASK 0
        Produces train, val, and test subsets for task 0

        train + val: 
            Extant_Leaves_minus_PNAS
        test:
            Extant_Leaves_in_PNAS
            
        Returns:
            task_0 (Dict[str,pd.DataFrame])

        """
        source_root_dir = self.source_root_dir
        A_minus_B_dir = Path(source_root_dir, "Extant_Leaves_family_10_512_minus_PNAS_family_100_512")
#         import pdb;pdb.set_trace()
#         pp(vars())
        config_path = list(A_minus_B_dir.glob("./CSVDataset-config.yaml"))[0]
        dataset_path = list(A_minus_B_dir.glob("./*full_dataset.csv"))[0]
        config = CSVDatasetConfig.load(path = config_path)
        dataset = CSVDataset.from_config(config, eager_encode_targets=False)
        ##########################################
        extant_minus_pnas_dataset = dataset
        A_minus_B_data_splits = DataSplitter.create_trainvaltest_splits(data=extant_minus_pnas_dataset,
                                                                        val_split=0.2,
                                                                        test_split="test",
                                                                        shuffle=True,
                                                                        seed=3654,
                                                                        stratify=True)

        test_config_path = A_minus_B_dir / "A_in_B-CSVDataset-config.yaml"
        test_config = CSVDatasetConfig.load(path = test_config_path)
        test_dataset = CSVDataset.from_config(test_config, eager_encode_targets=False)
        A_minus_B_data_splits['test'] = test_dataset
        task_0 = A_minus_B_data_splits
        return task_0


#     @staticmethod
    def setup_task_1(self): #experiment_root_dir = Path("/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0/")):
        """
        TASK 1
        Produces train, val, and test subsets for task 1

        train + val: 
            PNAS_minus_Extant_Leaves
        test:
            PNAS_in_Extant_Leaves
            
        Returns:
            task_1 (Dict[str,pd.DataFrame])


        """
        source_root_dir = self.source_root_dir
        B_minus_A_dir = Path(source_root_dir, "PNAS_family_100_512_minus_Extant_Leaves_family_10_512")
        config_path = list(B_minus_A_dir.glob("./CSVDataset-config.yaml"))[0]
        config = CSVDatasetConfig.load(path = config_path)
        dataset = CSVDataset.from_config(config)
        ##########################################
        pnas_minus_extant_dataset = dataset
        B_minus_A_data_splits = DataSplitter.create_trainvaltest_splits(data=pnas_minus_extant_dataset,
                                                                        val_split=0.2,
                                                                        test_split="test",
                                                                        shuffle=True,
                                                                        seed=3654,
                                                                        stratify=True)

        test_config_path = B_minus_A_dir / "A_in_B-CSVDataset-config.yaml"
        test_config = CSVDatasetConfig.load(path = test_config_path)
        test_dataset = CSVDataset.from_config(test_config)
        B_minus_A_data_splits['test'] = test_dataset
        task_1 = B_minus_A_data_splits

        return task_1
        
    def export_experiment_spec(self, output_root_dir=None):
        
        replace_class_indices = {"Nothofagaceae":"Fagaceae"}
        experiment_root_dir = output_root_dir or self.experiment_root_dir
        experiment_dir = Path(experiment_root_dir, self.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        print(f"Exporting experiment to experiment_dir: {experiment_dir}")
        
        task_0 = self.setup_task_0()
        task_1 = self.setup_task_1()

        task_0_label_encoder = task_0['train'].label_encoder
        task_0_label_encoder.__init__(replacements = replace_class_indices)

        task_0_label_encoder.fit(task_0['test'].targets)
#         print(len(task_0_label_encoder.classes))
        task_0_label_encoder.fit(task_0['train'].targets)
#         print(len(task_0_label_encoder.classes))

        task_0['val'].label_encoder = task_0_label_encoder
        task_0['test'].label_encoder = task_0_label_encoder

        task_0_dir = Path(experiment_dir, "task_0")
        os.makedirs(task_0_dir, exist_ok=True)
        for subset in ["train","val","test"]:
            task_0[subset].setup(samples_df=task_0[subset].samples_df,
                                 label_encoder=task_0[subset].label_encoder,
                                 fit_targets=False)

            CSVDatasetConfig.export_dataset_state(output_dir=task_0_dir, # / subset,
                                                  df=task_0[subset].samples_df,
                                                  config=task_0[subset].config,
                                                  encoder=task_0_label_encoder,
                                                  dataset_name=subset)

        task_1_label_encoder = task_1['train'].label_encoder
        task_1_label_encoder.__init__(replacements = replace_class_indices)
        task_1_label_encoder.fit(task_1['test'].targets)
        task_1_label_encoder.fit(task_1['train'].targets)
        task_1['val'].label_encoder = task_1_label_encoder
        task_1['test'].label_encoder = task_1_label_encoder

        task_1_dir = Path(experiment_dir, "task_1")
        os.makedirs(task_1_dir, exist_ok=True)
        for subset in ["train","val","test"]:
            task_1[subset].setup(samples_df=task_1[subset].samples_df,
                                 label_encoder=task_1[subset].label_encoder,
                                 fit_targets=False)
            CSVDatasetConfig.export_dataset_state(output_dir=task_1_dir, # / subset,
                                                  df=task_1[subset].samples_df,
                                                  config=task_1[subset].config,
                                                  encoder=task_1_label_encoder,
                                                  dataset_name=subset)
    
    
    
    def get_multitask_datasets(self,
                               train_transform=None,
                               train_target_transform=None,
                               val_transform=None,
                               val_target_transform=None) -> Tuple[Dict[str,CSVDataset]]:

        experiment_dir = Path(self.experiment_dir)
        task_0_dir = experiment_dir / "task_0"
        task_1_dir = experiment_dir / "task_1"
        task_0, task_1 = {}, {}

        for subset in ["train","val","test"]:
            task_0[subset], _ = CSVDatasetConfig.import_dataset_state(config_path = task_0_dir / f"{subset}.yaml")
            task_1[subset], _ = CSVDatasetConfig.import_dataset_state(config_path = task_1_dir / f"{subset}.yaml")

        if train_transform:
            task_0["train"].transform = train_transform
            task_1["train"].transform = train_transform
        if train_target_transform:
            task_0["train"].target_transform = train_target_transform
            task_1["train"].target_transform = train_target_transform

        for subset in ["val","test"]:
            if train_transform:
                task_0[subset].transform = val_transform
                task_1[subset].transform = val_transform
            if val_target_transform:
                task_0[subset].target_transform = val_target_transform
                task_1[subset].target_transform = val_target_transform

        return task_0, task_1


    
        
def cmdline_args():
    p = argparse.ArgumentParser(description="Export a series of dataset artifacts (containing csv catalog, yml config, json labels) for each dataset, provided that the corresponding images are pointed to by one of the file paths hard-coded in catalog_registry.py.")
    p.add_argument("-o", "--output_dir", dest="output_dir", type=str,
                   default="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0",
                   help="Output root directory. Each unique dataset will be allotted its own subdirectory within this root dir.")
    p.add_argument("-a", "--all", dest="make_all", action="store_true",
                   help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
    p.add_argument("-v", "--version", dest="version", type=str, default='v1_0',
                   help="Available dataset versions: [v0_3, v1_0].")
    
    return p.parse_args()

    
if __name__ == "__main__":
    
    args = cmdline_args()
    experiment = TransferExperiment()
    experiment.export_experiment_spec(output_root_dir=args.output_dir)