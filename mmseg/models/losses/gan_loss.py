import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODULES


@MODULES.register_module()
class GANLoss(nn.Module):
    def __init__(self,
                 loss_type='vanilla',
                 loss_name='loss_gan',
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        if loss_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, pred, label):
        loss = self.loss(pred, label)
        return loss * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
