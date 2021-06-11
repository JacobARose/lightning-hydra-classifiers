"""
lightning_hydra_classifiers/utils/callback_utils.py

Created by: Wednesday May 5th, 2021
Author: Jacob A Rose
"""



from pytorch_lightning.callbacks import Callback

class MyPrintingCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('Trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')
