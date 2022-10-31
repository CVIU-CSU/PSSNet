# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class UpSampeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(UpSampeBlock, self).__init__()
        self.conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, x, out=None):
        x = resize(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=False)
        if out is not None:
            x = torch.cat([x, out], 1)
        x = self.conv(x)
        return x


@HEADS.register_module()
class DenseUnetHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 arch = '121',
                 **kwargs):
        super(DenseUnetHead, self).__init__(**kwargs)
        self.arch = arch
        if arch == '121':
            self.up_1 = UpSampeBlock(1024 + 1024, 512, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_2 = UpSampeBlock(512 + 512, 256, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_3 = UpSampeBlock(256 + 256, 96, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_4 = UpSampeBlock(96 + 64, 96, self.conv_cfg, self.norm_cfg, self.act_cfg)
        if arch == '161':
            self.up_1 = UpSampeBlock(2208 + 2112, 768, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_2 = UpSampeBlock(768 + 768, 384, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_3 = UpSampeBlock(384 + 384, 96, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_4 = UpSampeBlock(96 + 96, 96, self.conv_cfg, self.norm_cfg, self.act_cfg)
        if arch == '201':
            self.up_1 = UpSampeBlock(1920 + 1792, 512, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_2 = UpSampeBlock(512 + 512, 256, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_3 = UpSampeBlock(256 + 256, 96, self.conv_cfg, self.norm_cfg, self.act_cfg)
            self.up_4 = UpSampeBlock(96 + 64, 96, self.conv_cfg, self.norm_cfg, self.act_cfg)
        # self.up_4 = UpSampeBlock(96, 96, self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.up_5 = UpSampeBlock(96, self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        conv1, x_64, x_32, x_16, out = x
        out = self.up_1(out, x_16)
        out = self.up_2(out, x_32)
        out = self.up_3(out, x_64)
        out = self.up_4(out, conv1)
        out = self.up_5(out)
        output = self.cls_seg(out)
        return output
