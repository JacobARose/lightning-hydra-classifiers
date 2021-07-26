"""
tests for pnas.py


Created by: Tuesday May 4th, 2021
Author: Jacob A Rose
"""


from contrastive_learning.data.pytorch.pnas import PNASLightningDataModule, PNASLeavesDataset
from contrastive_learning.data.pytorch.common import DataStageError, TrainValSplitDataset





class TestPNASLightningDataModule:
    
    def run(self):
        self.test_setup()
    

    def test_setup(self):
        
        data = PNASLightningDataModule()

        data.setup(stage='fit')
        assert hasattr(data, 'train_dataset')
        assert hasattr(data, 'val_dataset')
        assert not hasattr(data, 'test_dataset')

        assert isinstance(data.train_dataset, (PNASLeavesDataset,TrainValSplitDataset))
        assert isinstance(data.val_dataset, (PNASLeavesDataset,TrainValSplitDataset))
        del data.train_dataset
        del data.val_dataset

        ##########################################
        
        data.setup(stage='test')
        assert not hasattr(data, 'train_dataset')
        assert not hasattr(data, 'val_dataset')
        assert hasattr(data, 'test_dataset')
        assert isinstance(data.test_dataset, PNASLeavesDataset)
        del data.test_dataset
        
        ##########################################
        
        data.setup(stage=None)
        assert hasattr(data, 'train_dataset')
        assert hasattr(data, 'val_dataset')
        assert not hasattr(data, 'test_dataset')

        assert isinstance(data.train_dataset, (PNASLeavesDataset,TrainValSplitDataset))
        assert isinstance(data.val_dataset, (PNASLeavesDataset,TrainValSplitDataset))
        del data.train_dataset
        del data.val_dataset

        ##########################################
        
        try:
            data.setup(stage='other')
        except DataStageError as e:
            pass
            
        assert not hasattr(data, 'train_dataset')
        assert not hasattr(data, 'val_dataset')
        assert not hasattr(data, 'test_dataset')

        ##########################################
        ##########################################
        
        
        
        
        
if __name__ == "__main__":
    
    test = TestPNASLightningDataModule()
    
    test.run()
    
    from rich import print
    print(f'[100% SUCCESS] : {__file__}')