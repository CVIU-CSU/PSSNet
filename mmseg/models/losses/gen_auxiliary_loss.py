import torch.nn as nn

from ..builder import MODULES
from .cross_entropy_loss import CrossEntropyLoss
from .binary_loss import BinaryLoss


@MODULES.register_module()
class GenAuxLoss(nn.Module):
    def __init__(self,
                 loss_name='loss_gen_aux',
                 loss_weight=1.0):
        super(GenAuxLoss, self).__init__()
        self.loss = None  # todo
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        outputs_dict = args[0]
        pred = outputs_dict['aux_pred']
        label = outputs_dict['aux_label']
        label = label.squeeze(1)
        source = outputs_dict['source']
        if source == 'refuge':
            self.loss = CrossEntropyLoss(use_sigmoid=True, loss_weight=self.loss_weight)
        else:
            self.loss = BinaryLoss(loss_type='dice', smooth=1e-5, loss_weight=self.loss_weight)
        loss = self.loss(pred, label)
        if outputs_dict['with_auxiliary_head']:
            aux_pred = outputs_dict['aux_source_seg']
            loss += 0.1 * self.loss(aux_pred, label)
        return loss

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
