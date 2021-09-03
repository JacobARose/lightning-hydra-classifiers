"""

Transfer learning example on fake data.


Created on: Thursday, April 22nd, 2021
Author: Jacob A Rose


based on gist authored by jbschiratti [here](https://gist.github.com/jbschiratti/e93f1ff9cc518a93769101044160d64d)
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

EPOCH_MILESTONES = [5, 10]
NUM_CLASSES = 2


@torch.jit.script
def to_categorical(y_in: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-Hot encoding of target variables."""
    _encoding = torch.eye(num_classes)
    y_out = torch.zeros((y_in.size(0), num_classes))
    for j in range(y_in.size(0)):
        y_out[j] = _encoding[y_in[j]]
    return y_out


def _make_trainable(module):
    """Unfreeze a given module.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module, n=-1, train_bn=True):
    """Freeze the layers up to index n.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    n : int
        By default, all the layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.

    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            _recursive_freeze(module=child, train_bn=train_bn)
        else:
            _make_trainable(module=child)


def filter_params(module, train_bn=True):
    """Yield the trainable parameters of a given module.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    train_bn : bool (default: True)

    Returns
    -------
    generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            filter_params(module=child, train_bn=train_bn)


class TransferLearningModel(pl.LightningModule):
    """Transfer Learning with pre-trained ResNet50."""
    def __init__(self, hparams, train_dataset, val_dataset):
        super(TransferLearningModel, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.hparams = hparams
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained ResNet50:
        backbone = resnet50(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.features_extractor = torch.nn.Sequential(*_layers)
        freeze(module=self.features_extractor,
               train_bn=self.hparams.train_bn)

        # 2. Classif
        _mlp_layers = [torch.nn.Linear(2048, 256, bias=True),
                       torch.nn.Linear(256, 32, bias=True),
                       torch.nn.Linear(32, 2, bias=True)]
        self.fc = torch.nn.Sequential(*_mlp_layers)

        # 3. Loss
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        """Forward pass. Returns logits."""

        # Initially, x.shape = (B, 3, 224, 224).

        # 1. Feature extraction
        x = self.features_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        # 2. Classification
        x = self.fc(x)

        return x

    def loss(self, labels, logits):
        return self.loss_func(input=logits, target=labels)

    def train(self, mode=True):
        super(TransferLearningModel, self).train(mode=mode)

        # We want ensure that the feature extractor remains frozen (in eval
        # mode) as long as current_epoch < EPOCH_MILESTONES[0]:
        if self.current_epoch < self.hparams.epoch_milestones[0]:
            freeze(module=self.features_extractor,
                   train_bn=self.hparams.train_bn)

    def training_step(self, batch, batch_idx):

        # 1. Forward pass
        x, y = batch
        y_logits = self.forward(x)
        y_true = to_categorical(y, num_classes=NUM_CLASSES).type_as(x)

        # 2. Loss
        train_loss = self.loss(y_true, y_logits)

        # 3. Outputs
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})

        return output

    def training_epoch_end(self, outputs):

        train_loss_mean = 0.
        for output in outputs:

            train_loss = output['loss']
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                train_loss = torch.mean(train_loss)
            train_loss_mean += train_loss

        train_loss_mean /= len(outputs)
        return {'log': {'train_loss': train_loss_mean}}

    def validation_step(self, batch, batch_idx):

        # 1. Forward pass
        x, y = batch
        y_logits = self.forward(x)
        y_true = to_categorical(y, num_classes=NUM_CLASSES).type_as(x)

        # 2. Loss
        val_loss = self.loss(y_true, y_logits)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):

        val_loss_mean = 0.
        for output in outputs:

            val_loss = output['val_loss']
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        return {'log': {'val_loss': val_loss_mean}}

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                               lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.hparams.epoch_milestones,
                                                   gamma=0.1)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        """Train/val loaders."""

        loader = DataLoader(dataset=self.train_dataset if train else self.val_dataset,
                            batch_size=int(self.hparams.batch_size),
                            num_workers=self.hparams.num_workers,
                            shuffle=True if train else False)

        return loader

    def train_dataloader(self):
        log.info('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(train=False)


class UnfreezeCallback(Callback):
    """Unfreeze feature extractor callback."""

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == pl_module.hparams.epoch_milestones[0]:

            model = trainer.get_model()
            _make_trainable(model.features_extractor)

            optimizer = trainer.optimizers[0]
            _current_lr = optimizer.param_groups[0]['lr']
            optimizer.add_param_group({
                'params': filter_params(module=model.features_extractor,
                                        train_bn=pl_module.hparams.train_bn),
                'lr': _current_lr
            })


def main(hparams,
         train_dataset,
         val_dataset):
    # 1. Instantiate model
    model = TransferLearningModel(hparams,
                                  train_dataset=train_dataset,
                                  val_dataset=val_dataset)

    # 2. Setup trainer:
    # Train for exactly `hparams.nb_epochs` epochs
    trainer = pl.Trainer(
        weights_summary=None,
        show_progress_bar=True,
        num_sanity_val_steps=0,
        gpus=hparams.gpus,
        min_epochs=hparams.nb_epochs,
        max_epochs=hparams.nb_epochs,
        callbacks=[UnfreezeCallback()])

    trainer.fit(model)


if __name__ == '__main__':

    from argparse import Namespace

    import numpy as np
    from torch.utils.data import Subset
    from torchvision.transforms import ToTensor
    from torchvision.datasets import FakeData
    from sklearn.model_selection import train_test_split

    DATASET_SIZE = 112

    # 1. Create dataset and split into train/val
    train_idx, val_idx = train_test_split(np.arange(DATASET_SIZE),
                                          test_size=0.3,
                                          random_state=42)

    dataset = FakeData(size=DATASET_SIZE,
                       num_classes=NUM_CLASSES,
                       transform=ToTensor())
    training_dataset = Subset(dataset=dataset, indices=train_idx)
    validation_dataset = Subset(dataset=dataset, indices=val_idx)

    # 2. Define hparams
    _hparams = {'batch_size': 8,
                'num_workers': 6,
                'lr': 1e-2,
                'gpus': [0],
                'nb_epochs': 15,}

                