"""
contrastive_learning.data.pytorch.flash.callback.py

Based on the following commit for the file callback.py in PyTorch lightning-flash:
https://github.com/PyTorchLightning/lightning-flash/blob/6185bdabb504cb379151cd664a1422dc5fc44915/flash/data/callback.py

Added by: Jacob A Rose
Added on: Tuesday, April 14th 2021

"""

# from contextlib import contextmanager
# from typing import Any, List, Sequence

# from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.trainer.states import RunningStage
# from torch import Tensor

# import flash
# # from flash.data.utils import _STAGES_PREFIX
# _STAGES_PREFIX = {
#     RunningStage.TRAINING: 'train',
#     RunningStage.TESTING: 'test',
#     RunningStage.EVALUATING: 'val',
# #     RunningStage.VALIDATING: 'val',
#     RunningStage.PREDICTING: 'predict'
# } # from: https://github.com/PyTorchLightning/lightning-flash/blob/6185bdabb504cb379151cd664a1422dc5fc44915/flash/data/utils.py
# _STAGES_PREFIX_VALUES = {"train", "test", "val", "predict"}



# class FlashCallback(Callback):

#     def on_load_sample(self, sample: Any, running_stage: RunningStage) -> None:
#         """Called once a sample has been loaded using ``load_sample``."""

#     def on_pre_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
#         """Called once ``pre_tensor_transform`` has been applied to a sample."""

#     def on_to_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
#         """Called once ``to_tensor_transform`` has been applied to a sample."""

#     def on_post_tensor_transform(self, sample: Tensor, running_stage: RunningStage) -> None:
#         """Called once ``post_tensor_transform`` has been applied to a sample."""

#     def on_per_batch_transform(self, batch: Any, running_stage: RunningStage) -> None:
#         """Called once ``per_batch_transform`` has been applied to a batch."""

#     def on_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
#         """Called once ``collate`` has been applied to a sequence of samples."""

#     def on_per_sample_transform_on_device(self, sample: Any, running_stage: RunningStage) -> None:
#         """Called once ``per_sample_transform_on_device`` has been applied to a sample."""

#     def on_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
#         """Called once ``per_batch_transform_on_device`` has been applied to a sample."""


# class ControlFlow(FlashCallback):

#     def __init__(self, callbacks: List[FlashCallback]):
#         self._callbacks = callbacks

#     def run_for_all_callbacks(self, *args, method_name: str, **kwargs):
#         if self._callbacks:
#             for cb in self._callbacks:
#                 getattr(cb, method_name)(*args, **kwargs)

#     def on_load_sample(self, sample: Any, running_stage: RunningStage) -> None:
#         self.run_for_all_callbacks(sample, running_stage, method_name="on_load_sample")

#     def on_pre_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
#         self.run_for_all_callbacks(sample, running_stage, method_name="on_pre_tensor_transform")

#     def on_to_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
#         self.run_for_all_callbacks(sample, running_stage, method_name="on_to_tensor_transform")

#     def on_post_tensor_transform(self, sample: Tensor, running_stage: RunningStage) -> None:
#         self.run_for_all_callbacks(sample, running_stage, method_name="on_post_tensor_transform")

#     def on_per_batch_transform(self, batch: Any, running_stage: RunningStage) -> None:
#         self.run_for_all_callbacks(batch, running_stage, method_name="on_per_batch_transform")

#     def on_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
#         self.run_for_all_callbacks(batch, running_stage, method_name="on_collate")

#     def on_per_sample_transform_on_device(self, sample: Any, running_stage: RunningStage) -> None:
#         self.run_for_all_callbacks(sample, running_stage, method_name="on_per_sample_transform_on_device")

#     def on_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
#         self.run_for_all_callbacks(batch, running_stage, method_name="on_per_batch_transform_on_device")


# class BaseDataFetcher(FlashCallback):
#     """
#     This class is used to profile :class:`~flash.data.process.Preprocess` hook outputs.
#     By default, the callback won't profile the data being processed as it may lead to ``OOMError``.
#     Example::
#         from flash.data.callback import BaseDataFetcher
#         from flash.data.data_module import DataModule
#         class PrintData(BaseDataFetcher):
#             def print(self):
#                 print(self.batches)
#         class CustomDataModule(DataModule):
#             @staticmethod
#             def configure_data_fetcher():
#                 return PrintData()
#             @classmethod
#             def from_inputs(
#                 cls,
#                 train_data: Any,
#                 val_data: Any,
#                 test_data: Any,
#                 predict_data: Any) -> "CustomDataModule":
#                 preprocess = cls.preprocess_cls()
#                 return cls.from_load_data_inputs(
#                     train_load_data_input=train_data,
#                     val_load_data_input=val_data,
#                     test_load_data_input=test_data,
#                     predict_load_data_input=predict_data,
#                     preprocess=preprocess,
#                     batch_size=5)
#         dm = CustomDataModule.from_inputs(range(5), range(5), range(5), range(5))
#         data_fetcher = dm.data_fetcher
#         # By default, the ``data_fetcher`` is disabled to prevent OOM.
#         # The ``enable`` context manager will activate it.
#         with data_fetcher.enable():
#             # This will fetch the first val dataloader batch.
#             _ = next(iter(dm.val_dataloader()))
#         data_fetcher.print()
#         # out:
#         {
#             'train': {},
#             'test': {},
#             'val': {
#                 'load_sample': [0, 1, 2, 3, 4],
#                 'pre_tensor_transform': [0, 1, 2, 3, 4],
#                 'to_tensor_transform': [0, 1, 2, 3, 4],
#                 'post_tensor_transform': [0, 1, 2, 3, 4],
#                 'collate': [tensor([0, 1, 2, 3, 4])],
#                 'per_batch_transform': [tensor([0, 1, 2, 3, 4])]},
#             'predict': {}
#         }
#         data_fetcher.reset()
#         data_fetcher.print()
#         # out:
#         {
#             'train': {},
#             'test': {},
#             'val': {},
#             'predict': {}
#         }
#     """

#     def __init__(self, enabled: bool = False):
#         self.enabled = enabled
#         self._preprocess = None
#         self.reset()

#     def _store(self, data: Any, fn_name: str, running_stage: RunningStage) -> None:
#         if self.enabled:
#             store = self.batches[_STAGES_PREFIX[running_stage]]
#             store.setdefault(fn_name, [])
#             store[fn_name].append(data)

#     def on_load_sample(self, sample: Any, running_stage: RunningStage) -> None:
#         self._store(sample, "load_sample", running_stage)

#     def on_pre_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
#         self._store(sample, "pre_tensor_transform", running_stage)

#     def on_to_tensor_transform(self, sample: Any, running_stage: RunningStage) -> None:
#         self._store(sample, "to_tensor_transform", running_stage)

#     def on_post_tensor_transform(self, sample: Tensor, running_stage: RunningStage) -> None:
#         self._store(sample, "post_tensor_transform", running_stage)

#     def on_per_batch_transform(self, batch: Any, running_stage: RunningStage) -> None:
#         self._store(batch, "per_batch_transform", running_stage)

#     def on_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
#         self._store(batch, "collate", running_stage)

#     def on_per_sample_transform_on_device(self, samples: Sequence, running_stage: RunningStage) -> None:
#         self._store(samples, "per_sample_transform_on_device", running_stage)

#     def on_per_batch_transform_on_device(self, batch: Any, running_stage: RunningStage) -> None:
#         self._store(batch, "per_batch_transform_on_device", running_stage)

#     @contextmanager
#     def enable(self):
#         """This function is used to enable to BaseDataFetcher"""
#         self.enabled = True
#         yield
#         self.enabled = False

#     def attach_to_datamodule(self, datamodule) -> None:
#         datamodule.data_fetcher = self

#     def attach_to_preprocess(self, preprocess: 'flash.data.process.Preprocess') -> None:
#         preprocess.add_callbacks([self])
#         self._preprocess = preprocess

#     def reset(self):
#         self.batches = {k: {} for k in _STAGES_PREFIX.values()}