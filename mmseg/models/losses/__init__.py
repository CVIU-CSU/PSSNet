# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .binary_loss import BinaryLoss
from .gan_loss import GANLoss
from .gen_auxiliary_loss import GenAuxLoss
from .multi_gen_auxiliary_loss import MultiGenAuxLoss
from .kl_loss import KLLoss
from .class_balanced_cross_entropy_loss import ClassBalancedCrossEntropyLoss
from .focal_loss import FocalLoss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss', 'BinaryLoss',
]
