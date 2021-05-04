




# resnet50 = models.resnet50(pretrained=True)
# for param in resnet50.parameters():
#     param.requires_grad = False
# num_classes = 10
# resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)
# preds = softmax(preds, dim=-1)
# pred_labels = torch.argmax(preds, dim=-1)
# pred_labels[:5]


#########################################
#########################################

from torch import nn
from torch.nn.functional import softmax
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from torch.nn.functional import cross_entropy
from torch.optim import Adam




class CNNPretrainedModel(nn.Module):
    """
    Customized from fastai learner
    """
    def __init__(self, base_arch, no_classes, dropout=0.5, init=nn.init.kaiming_normal_):
        super(CNNPretrainedModel, self).__init__()

        self.model = create_cnn_model(base_arch, no_classes, ps=dropout)
        self.meta = cnn_config(base_arch)
        self.split(self.meta['split'])
        self.freeze()

        apply_init(self.model[1], init)

    def split(self, split_on):
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on(self.model)
        self.layer_groups = split_model(self.model, split_on)
        return self

    def freeze_to(self, n):
        "Freeze layers up to layer group `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not isinstance(l, bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)

    def freeze(self):
        "Freeze up to last layer group."
        assert(len(self.layer_groups) > 1)
        self.freeze_to(-1)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def forward(self, x):
        return self.model.forward(x)

#########################################
#########################################






class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # self.num_classes = num_classes
        # self.lr = lr

        self.model = models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def training_step(self, batch, batch_idx):
        # return the loss given a batch: this has a computational graph attached to it: optimization
        x, y = batch
        preds = self.model(x)
        loss = cross_entropy(preds, y)
        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value
        self.log('train_acc', accuracy(preds, y))
        return loss

    def configure_optimizers(self):
        # return optimizer
        optimizer = Adam(self.model.fc.parameters(), lr=self.hparams.lr)
        return optimizer








class SimpleDenseNet(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["lin1_size"]),
            nn.BatchNorm1d(hparams["lin1_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin1_size"], hparams["lin2_size"]),
            nn.BatchNorm1d(hparams["lin2_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin2_size"], hparams["lin3_size"]),
            nn.BatchNorm1d(hparams["lin3_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin3_size"], hparams["output_size"]),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)
