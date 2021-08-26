"""

lightning_hydra_classifiers/data/utils/catalog_registry.py

Author: Jacob A Rose
Created: Sunday August 1st, 2021

Importable registry of directories containing different datasets and versions mounted on data_cifs.

Currently covers:
    - leavesdb v0_3
    
Work In Progress:
    - leavesdb v1_0



python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/catalog_registry.py"


"""

import argparse
from dataclasses import dataclass
import rich
from rich import print as pp
from typing import *


__all__ = ["leavesdbv0_3", "leavesdbv1_0", "available_datasets"]



    
    
    
@dataclass
class LeavesdbBase:

#     datasets = ["PNAS", "Extant", "Fossil"]
    
    def keys(self):
        return self.__dict__.keys()
        
    def __getitem__(self, index):
        return self.__dict__[index]
    

    @property
    def datasets(self):
        return {"PNAS":self.PNAS,
                "Extant":self.Extant,
                "Fossil":self.Fossil}
    
    @property
    def tags(self):
        out = {}
        for k,v in self.datasets.items():
            out[k] = []
            for tag in v.keys():
                out[k].append(tag)
        return out
    
    def __repr__(self):
        return rich.pretty.pretty_repr(self.datasets)
    
    @property
    def PNAS(self):
        return {k:self[k] for k in self.keys() if k.startswith("PNAS")}
    
    @property
    def Extant(self):
        return {k:self[k] for k in self.keys() if k.startswith("Extant")}
    
    @property
    def Fossil(self):
        return {k:self[k] for k in self.keys() if k.startswith("Fossil")}
    
    
    
    


@dataclass
class Leavesdbv0_3(LeavesdbBase):

    PNAS_family_100_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100"
    PNAS_family_4_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_4"
    
    PNAS_family_100_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_512"
    PNAS_family_100_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1024"
    PNAS_family_100_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1536"
    PNAS_family_100_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_2048"
################################S
################################
    Extant_family_10_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/512"
    Extant_family_10_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/1024"
    Extant_family_10_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/1536"
    Extant_family_10_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/2048"
    Extant_family_20_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/512"
    Extant_family_20_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/1024"
    Extant_family_20_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/1536"
    Extant_family_20_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/2048"
    Extant_family_50_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/512"
    Extant_family_50_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/1024"
    Extant_family_50_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/1536"
    Extant_family_50_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/2048"
    Extant_family_100_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/512"
    Extant_family_100_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/1024"
    Extant_family_100_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/1536"
    Extant_family_100_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/2048"
################################
################################
    Wilf_Fossil_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Wilf_Fossil",
    Wilf_Fossil_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Wilf_Fossil",
    Wilf_Fossil_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Wilf_Fossil",
    Wilf_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Wilf_Fossil",
    Florissant_Fossil_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Florissant_Fossil",
    Florissant_Fossil_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Florissant_Fossil",
    Florissant_Fossil_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Florissant_Fossil",
    Florissant_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Florissant_Fossil"
################################
################################
    Fossil_512: List[str] = None
    Fossil_1024: List[str] = None
    Fossil_1536: List[str] = None
    Fossil_2048: List[str] = None

    def __post_init__(self):
        
        self.Fossil_512: List[str] = ["/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Wilf_Fossil",
                                 "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Florissant_Fossil"]
        self.Fossil_1024: List[str] = ["/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Wilf_Fossil",
                                  "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Florissant_Fossil"]
        self.Fossil_1536: List[str] = ["/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Wilf_Fossil",
                                  "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Florissant_Fossil"]
        self.Fossil_2048: List[str] = ["/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Wilf_Fossil",
                                  "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Florissant_Fossil"]
            
    @property
    def datasets(self):
        return {"PNAS":self.PNAS,
                "Extant":self.Extant,
                "Fossil":self.Fossil}
            
    def __repr__(self):
        return rich.pretty.pretty_repr(self.datasets)


@dataclass
class Leavesdbv1_0(LeavesdbBase):

    Extant_Leaves_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/original/full/jpg"
    General_Fossil_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/original/full/jpg"
    Florissant_Fossil_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/original/full/jpg"

    
    PNAS_family_100_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_512"
    PNAS_family_100_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1024"
    PNAS_family_100_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1536"
    PNAS_family_100_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_2048"
# ################################S
# ################################
    Extant_family_10_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/10/jpg"
    Extant_family_10_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/10/jpg"
    Extant_family_10_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/10/jpg"
    Extant_family_10_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/10/jpg"
    Extant_family_20_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/20/jpg"
    Extant_family_20_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/20/jpg"
    Extant_family_20_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/20/jpg"
    Extant_family_20_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/20/jpg"
    Extant_family_10_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/50/jpg"
    Extant_family_50_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/50/jpg"
    Extant_family_50_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/50/jpg"
    Extant_family_50_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/50/jpg"
    Extant_family_100_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/100/jpg"
    Extant_family_100_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/100/jpg"
    Extant_family_100_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/100/jpg"
    Extant_family_100_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/100/jpg"

# ################################
# ################################

    General_Fossil_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/512/full/jpg"
    General_Fossil_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/1024/full/jpg"
    General_Fossil_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/1536/full/jpg"
    General_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/2048/full/jpg"
    Florissant_Fossil_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/512/full/jpg"
    Florissant_Fossil_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/1024/full/jpg"
    Florissant_Fossil_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/1536/full/jpg"
    Florissant_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/2048/full/jpg"


    General_Fossil_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/512/3/jpg"
    General_Fossil_family_3_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/1024/3/jpg"
    General_Fossil_family_3_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/1536/3/jpg"
    General_Fossil_family_3_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/2048/3/jpg"
    Florissant_Fossil_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/512/3/jpg"
    Florissant_Fossil_family_3_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/1024/3/jpg"
    Florissant_Fossil_family_3_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/1536/3/jpg"
    Florissant_Fossil_family_3_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/2048/3/jpg"

# ################################
# ################################
    Fossil_512: List[str] = None
    Fossil_1024: List[str] = None
    Fossil_1536: List[str] = None
    Fossil_2048: List[str] = None
    Fossil_family_3_512: List[str] = None
    Fossil_family_3_1024: List[str] = None
    Fossil_family_3_1536: List[str] = None
    Fossil_family_3_2048: List[str] = None

    def __post_init__(self):
        
        self.Fossil_original: List[str] = [self.General_Fossil_original,
                                           self.Florissant_Fossil_original]

        self.Fossil_512: List[str] = [self.General_Fossil_512,
                                      self.Florissant_Fossil_512]
        self.Fossil_1024: List[str] = [self.General_Fossil_1024,
                                      self.Florissant_Fossil_1024]
        self.Fossil_1536: List[str] = [self.General_Fossil_1536,
                                      self.Florissant_Fossil_1536]
        self.Fossil_2048: List[str] = [self.General_Fossil_2048,
                                      self.Florissant_Fossil_2048]
        self.Fossil_family_3_512: List[str] = [self.General_Fossil_family_3_512,
                                               self.Florissant_Fossil_family_3_512]
        self.Fossil_family_3_1024: List[str] = [self.General_Fossil_family_3_1024,
                                                self.Florissant_Fossil_family_3_1024]
        self.Fossil_family_3_1536: List[str] = [self.General_Fossil_family_3_1536,
                                                self.Florissant_Fossil_family_3_1536]
        self.Fossil_family_3_2048: List[str] = [self.General_Fossil_family_3_2048,
                                                self.Florissant_Fossil_family_3_2048]


    @property
    def datasets(self):
        return {"PNAS":self.PNAS,
                "Extant":self.Extant,
                "Fossil":self.Fossil}

    @property
    def tags(self):
        out = {}
        for k,v in self.datasets.items():
            out[k] = []
            for tag in v.keys():
                out[k].append(tag)
        return out
        
    
    def __repr__(self):
        return rich.pretty.pretty_repr(self.datasets)



# @dataclass
# class available_datasets:
    
#     versions = ["v0_3", "v1_0"]
    
#     def __getitem__(self, index):
#         return self.versions[index]

#     def __repr__(self):

leavesdbv0_3 = Leavesdbv0_3()
leavesdbv1_0 = Leavesdbv1_0()



class available_datasets:
    versions = {"v0_3": leavesdbv0_3,
                "v1_0": leavesdbv1_0}
    
    def __repr__(self) -> str:
        out = ""
        for k,v in self.versions.items():
            out += f"{k}:" + "\n" + v.__repr__() + "\n"
        return out
    
    @classmethod
    def query_tags(cls,
                   dataset_name: str,
                   threshold: Optional[int]=0,
                   y_col: str="family", 
                   resolution: Optional[Tuple[str, int]]="original") -> str:
        tag = dataset_name
        if int(threshold) > 0:
            tag += f"_{y_col}_{threshold}"
        tag += f"_{resolution}"
        
        try:
            cls.get_latest(tag)
        except KeyError as e:
            print(f"KeyError: Invalid dataset query. {tag} doesn't exist")
            print("tag: ", tag)
        return tag
#         if isinstance(resolution, int):
            
    
    @classmethod
    def get_latest(cls, tag: str) -> Union[str, List[str]]:
        """
        Wrapper around available_datasets.get() that defaults to the latest dataset version containing the requested tag.
        
        Useful for datasets like PNAS, which havent changed since version 0_3.
        
        """
        if "PNAS" in tag:
            return cls.get(tag, version="v0_3")
        else:
            return cls.get(tag, version="v1_0")

    @classmethod
    def get(cls, tag: str, version: str="v1_0") -> Union[str, List[str]]:
        """
        Examples:
        ---------
        available_datasets.get(tag='Fossil_2048', version='v1_0')
        >>['/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/2048/full/jpg',
           '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/2048/full/jpg']
           
        available_datasets.get(tag='Extant_family_20_1024', version='v1_0')
        >>'/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/20/jpg'

        """
        return cls.versions[version][tag]


    
    @property
    def tags(self):
        out = {}
        for version,dataset in self.versions.items():
            out[version] = dataset.tags
        return out




        
def cmdline_args():
    p = argparse.ArgumentParser(description="catalog_registry.py -- Module containing key: value mappings between accepted dataset names with versions, and their corresponding locations on data_cifs.")
    p.add_argument("-t", "--tags", action="store_true",
                   help="User provides this flag to display a concise summary of available data, containing all tag info while omitting paths.")
    p.add_argument("-d", "--display", action="store_true",
                   help="User provides this flag to display a full listing of all datasets with versions, mapped to their expected data_cifs locations.")
    return p.parse_args()








if __name__ == "__main__":
    
    args = cmdline_args()
    
    if args.tags:
        pp(available_datasets().tags)
    elif args.display:
        pp(available_datasets())
    else:
        print(f'Provide either --tags or --diplay if running catalog_registry.py from the command line.')

        