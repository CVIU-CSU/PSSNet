# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..losses import accuracy


@HEADS.register_module()
class FCNHeadGroup(BaseDecodeHead):

    def __init__(self,
                 num_classes_multi,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 groups=3,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.groups = groups
        super(FCNHeadGroup, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels * self.groups,
                self.channels * self.groups,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False,
                groups=self.groups))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels * self.groups,
                    self.channels * self.groups,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    groups=self.groups))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                (self.in_channels + self.channels) * self.groups,
                self.channels * self.groups,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                groups=self.groups)
        self.conv_seg = nn.ModuleList([self.conv_seg])
        if not isinstance(num_classes_multi, list):
            num_classes_multi = [num_classes_multi]
        for cla in num_classes_multi:
            self.conv_seg.append(nn.Conv2d(self.channels, cla, kernel_size=1))
        self.num_classes = [self.num_classes] + num_classes_multi

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:   # False
            output = self.conv_cat(torch.cat([x, output], dim=1))
        N, C, H, W = output.shape
        output = output.reshape(N, self.groups, C // self.groups, H, W).permute(1, 0, 2, 3, 4)
        if self.dropout is not None:
            output = self.dropout(output)
        outputs = []
        for i, cs in enumerate(self.conv_seg):
            outputs.append(cs(output[i]))
        return outputs

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, img_metas):
        """Compute segmentation loss."""
        loss = dict()
        for i in range(len(seg_logit)):
            seg_logit[i] = resize(
                input=seg_logit[i],
                size=seg_label.shape[2:],
                mode='bilinear')
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        num_task = len(self.num_classes)
        num_class = torch.cumsum(torch.tensor(self.num_classes), dim=0)
        num_img = seg_label.shape[0]
        batch_size_per_task = num_img // num_task
        for i in range(num_task):  # number of task
            idx = batch_size_per_task * i
            per_seg_logit = seg_logit[i]
            per_seg_label = seg_label[idx:(idx + batch_size_per_task)]
            per_img_metas = img_metas[idx:(idx + batch_size_per_task)]
            loss[f'loss_task{i}'] = self.loss_decode[i](
                per_seg_logit,
                per_seg_label,
                img_metas=per_img_metas,
                weight=seg_weight,
                ignore_index=self.ignore_index)

            loss[f'acc_seg{i}'] = accuracy(per_seg_logit, per_seg_label)
        return loss
