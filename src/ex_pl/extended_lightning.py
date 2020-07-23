import collections
import inspect
import os
import re
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.distributed as torch_distrib
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.core.grads import GradInformation
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.saving import ModelIO, PRIMITIVE_TYPES, ALLOWED_CONFIG_TYPES
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.parsing import AttributeDict, collect_init_args, get_init_args
from pytorch_lightning.core.lightning import LightningModule


try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class ExtendedLightningModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def qual_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        r""" see the actual docs"""

    def qual_step_end(self, *args, **kwargs) -> Dict[str, Tensor]:
        """ see the actual docs"""
    
    def qual_end(self, outputs):
        """ deprecated """

    def qual_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        """ I will not write all this out """
    
    def qual_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """ see the docs"""
