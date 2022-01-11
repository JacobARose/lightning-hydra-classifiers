

import pytorch_lightning as pl
from dataclasses import dataclass

    """
    A dataclass for producing a default configuration for instantiating a pytorch lightning module with multiple classifiers for multiple tasks
    """

    @dataclass
    class MultiTaskModelConfig: