
import hydra


import os
import shutil


# Source: https://github.com/Erlemar/wheat/blob/master/src/utils/utils.py
def save_useful_info():
    shutil.copytree(os.path.join(hydra.utils.get_original_cwd(), 'lightning_hydra_classifiers'), os.path.join(os.getcwd(), 'code/lightning_hydra_classifiers'))
    shutil.copy2(os.path.join(hydra.utils.get_original_cwd(), 'hydra_run.py'), os.path.join(os.getcwd(), 'code'))