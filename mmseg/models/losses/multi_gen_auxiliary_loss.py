import torch.nn as nn

from ..builder import MODULES
from ..builder import build_loss


@MODULES.register_module()
class MultiGenAuxLoss(nn.Module):
    def __init__(self,
                 loss_name='loss_gen_task',
                 loss_decode=[
                     dict(
                         type='BinaryLoss',
                         loss_type='dice',
                         smooth=1e-5,
                         loss_weight=1.0),
                     dict(
                         type='BinaryLoss',
                         loss_type='dice',
                         smooth=1e-5,
                         loss_weight=1.0),
                 ],
                 loss_weight=1.0):
        super(MultiGenAuxLoss, self).__init__()
        self.loss_decode = nn.ModuleList()
        if isinstance(loss_decode, dict):
            self.loss_decode.append(build_loss(loss_decode))
        elif isinstance(loss_decode, (list, tuple)):
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        self._loss_name = loss_name
        self.loss_weight = loss_weight

    def forward(self, aux_pred, aux_label, aux_source_seg, with_auxiliary_head):
        loss = dict()
        num = len(aux_pred)
        for i in range(num):
            pred = aux_pred[i]
            label = aux_label[i]
            label = label.squeeze(1)
            loss[f'loss_gen_task_{i+1}'] = self.loss_decode[i](pred, label)
            if with_auxiliary_head:
                auxi_pred = aux_source_seg[i]
                loss[f'loss_gen_task_aux_{i+1}'] = 0.4 * self.loss_decode[i](auxi_pred, label)
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
