import time
import warnings

import mmcv
import torch

from mmcv.runner import RUNNERS
from .multi_task_iterbased_runner import MultiTaskIterBasedRunner


@RUNNERS.register_module()
class MultiTaskSingleIterBasedRunner(MultiTaskIterBasedRunner):

    def __init__(self, **kwargs):
        super(MultiTaskSingleIterBasedRunner, self).__init__(**kwargs)
        self.task_idx = 0

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader[0]
        self._epoch = data_loader[0].epoch
        self.task_idx = (self.task_idx + 1) % len(data_loader)
        data_batch = []
        for _ in range(len(data_loader)):
            data_batch.append(next(data_loader[self.task_idx]))
        kwargs['task_id'] = self.task_idx

        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1


