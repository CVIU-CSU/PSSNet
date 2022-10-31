# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, trunc_normal_init, build_norm_layer
from mmcv.utils import to_2tuple
from mmcv.runner import force_fp32, BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import build_dropout, FFN
from mmcv.cnn.utils.weight_init import trunc_normal_

from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..losses import accuracy


class WindowMSA(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        trunc_normal_init(self.proj, std=0.02)
        trunc_normal_init(self.q, std=0.02)
        trunc_normal_init(self.kv, std=0.02)

    def forward(self, feat1, feat2, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N1, C1 = feat1.shape
        _, N2, C2 = feat2.shape
        feat1 = self.norm1(feat1)
        feat2 = self.norm1(feat2)
        q = self.q(feat1).reshape(B, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(feat2).reshape(B, N2, 2, self.num_heads,
                                  C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N1,
                             N2) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N1, N2)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):

    def __init__(self,
                 groups,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.groups = groups
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = ModuleList()
        self.norm2 = nn.ModuleList()
        self.ffn = ModuleList()

        self.drop = build_dropout(dropout_layer)

        self.proj = ConvModule(
                embed_dims * (self.groups - 1),
                embed_dims,
                1,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
                act_cfg=dict(type='ReLU'))
        
        for i in range(self.groups):
            for j in range(self.groups - 1):
                self.w_msa.append(WindowMSA(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    window_size=to_2tuple(window_size),
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop_rate=attn_drop_rate,
                    proj_drop_rate=proj_drop_rate,
                    init_cfg=None))
                self.norm2.append(build_norm_layer(norm_cfg, embed_dims)[1])
                self.ffn.append(FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * 4,
                    num_fcs=2,
                    ffn_drop=proj_drop_rate,
                    dropout_layer=dropout_layer,
                    act_cfg=act_cfg,
                    add_identity=True,
                    init_cfg=None))


    def forward(self, output):
        _, Bx, Cx, Hx, Wx = output.shape
        output = output.flatten(3).transpose(2, 3)
        N = Hx * Wx
        output_window = []
        for i in range(self.groups):
            query = output[i]
            B, L, C = query.shape
            query = query.view(B, Hx, Wx, C)

            # pad feature maps to multiples of window size
            pad_r = (self.window_size - Wx % self.window_size) % self.window_size
            pad_b = (self.window_size - Hx % self.window_size) % self.window_size
            query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
            H_pad, W_pad = query.shape[1], query.shape[2]
            query_windows = self.window_partition(query)
            # nW*B, window_size*window_size, C
            query_windows = query_windows.view(-1, self.window_size ** 2, C)
            B, N, C = query_windows.shape
            output_window.append(query_windows)

        outputs = []
        idx = 0
        for i in range(self.groups):
            tmp = []
            for j in range(self.groups):
                if j == i:
                    continue
                # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
                attn_windows = self.w_msa[idx](output_window[i], output_window[j], mask=None)

                # merge windows
                attn_windows = attn_windows.view(-1, self.window_size,
                                                 self.window_size, C)

                # B H' W' C
                shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
                x = shifted_x

                if pad_r > 0 or pad_b:
                    x = x[:, :Hx, :Wx, :].contiguous()

                x = x.view(Bx, Hx * Wx, Cx)

                x = self.drop(x)

                x = x + output[i]

                identity = x
                x = self.norm2[idx](x)
                x = self.ffn[idx](x, identity=identity)

                x = x.transpose(1, 2).view(Bx, Cx, Hx, Wx)
                tmp.append(x)
            tmp = torch.cat(tmp, dim=1)
            outputs.append(self.proj(tmp) + output[i].transpose(1, 2).view(Bx, Cx, Hx, Wx))
            # attn_feat = torch.cat(tmp, dim=2)
            # output.append(self.proj(attn_feat).transpose(1, 2).view(Bx, Cx, Hx, Wx))
        return outputs

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


@HEADS.register_module()
class UperHeadGroup(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, num_classes_multi, pool_scales=(1, 2, 3, 6), groups=3, attn=True, qkv_bias=True,
                 num_heads=4, window_size=8, qk_scale=None, attn_drop_rate=0.0, proj_drop_rate=0.0,
                 drop_path_rate=0.0, **kwargs):
        super(UperHeadGroup, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.groups = groups
        self.attn = attn
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1] * self.groups,
            self.channels * self.groups,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
            groups=groups)
        self.bottleneck = ConvModule(
            (self.in_channels[-1] + len(pool_scales) * self.channels) * self.groups,
            self.channels * self.groups,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            groups=groups)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels * self.groups,
                self.channels * self.groups,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False,
                groups=groups)
            fpn_conv = ConvModule(
                self.channels * self.groups,
                self.channels * self.groups,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False,
                groups=groups)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            (len(self.in_channels) * self.channels) * self.groups,
            self.channels * self.groups,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            groups=groups)
        self.conv_seg = nn.ModuleList([self.conv_seg])
        if not isinstance(num_classes_multi, list):
            num_classes_multi = [num_classes_multi]
        for cla in num_classes_multi:
            self.conv_seg.append(nn.Conv2d(self.channels, cla, kernel_size=1))
        self.num_classes = [self.num_classes] + num_classes_multi

        embed_dims = self.channels

        if attn:
            self.attn = ShiftWindowMSA(
                groups=self.groups,
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=proj_drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                init_cfg=None)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        ppm_outs = self.psp_modules(x)
        num_pool = len(ppm_outs)
        ppm_outs = torch.cat(ppm_outs, dim=0)
        N, C, H, W = ppm_outs.shape
        ppm_outs = ppm_outs.reshape(N, self.groups, C//self.groups, H, W).permute(1, 0, 2, 3, 4)
        Nx, Cx, Hx, Wx = x.shape
        x = x.reshape(Nx, self.groups, Cx // self.groups, Hx, Wx).permute(1, 0, 2, 3, 4)
        psp_outs = []
        for i in range(self.groups):
            psp_outs.append(x[i])
            psp_outs.append(ppm_outs[i].reshape(N//num_pool, num_pool*C//self.groups, H, W))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs, **kwargs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # laterals.append(self.bottleneck(inputs[-1]))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        num_fpn = len(fpn_outs)
        fpn_outs = torch.cat(fpn_outs, dim=0)
        N, C, H, W = fpn_outs.shape
        fpn_outs = fpn_outs.reshape(N, self.groups, C // self.groups, H, W).permute(1, 0, 2, 3, 4)
        outs = []
        for i in range(self.groups):
            outs.append(fpn_outs[i].reshape(N // num_fpn, num_fpn * C // self.groups, H, W))
        fpn_outs = torch.cat(outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        N, C, H, W = output.shape
        output = output.reshape(N, self.groups, C // self.groups, H, W).permute(1, 0, 2, 3, 4)
        if self.dropout is not None:
            output = self.dropout(output)

        if self.attn:
	        if hasattr(kwargs, 'semi') and kwargs[semi]:
	            pass
	        else:
	            output = self.attn(output)
        outputs = []
        for i, cs in enumerate(self.conv_seg):
            outputs.append(cs(output[i]))
        # outputs = torch.cat(outputs, dim=1)
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
        # noinspection PyInterpreter
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
