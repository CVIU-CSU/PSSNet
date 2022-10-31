from ..builder import MODULES
from .encoder_decoder import EncoderDecoder
from .encoder_multi_decoder import Encoder_Multi_Decoder
from .encoder_decoder_multi_task_group import Encoder_Decoder_Multi_task_group

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from mmseg.ops import resize


@MODULES.register_module()
class Generator(EncoderDecoder):
    def __init__(self, use_sigmoid=False, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.use_sigmoid = use_sigmoid

    def encode_decode(self, img, img_metas, with_auxiliary_head=False):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if with_auxiliary_head:
            auxilary_out = self.auxiliary_head.forward_test(x, img_metas, self.test_cfg)
            auxilary_out = resize(
                input=auxilary_out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            out = torch.cat((out, auxilary_out))
        return out

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_meta, return_loss=False, with_auxiliary_head=False, source='refuge', **kwargs):
        seg_logit = self.encode_decode(img, img_meta, with_auxiliary_head=with_auxiliary_head)
        if not return_loss:  # discriminator
            size = (512, 512)
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        if source == 'refuge':
            output = F.softmax(seg_logit, dim=1)
        else:
            output = torch.sigmoid(seg_logit)
        return seg_logit, output

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if self.use_sigmoid:
            output = torch.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, test_mode=False, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if self.use_sigmoid:
            seg_logit = seg_logit.squeeze(0).cpu().numpy()
            if test_mode:
                return [(seg_logit, img_meta[0]['ori_filename'].split('.')[0])]
            seg_logit = [(seg_logit, True, True)]
            return seg_logit
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        if test_mode:
            seg_pred.append(img_meta[0]['ori_filename'].split('.')[0])
            return [seg_pred]
        else:
            return seg_pred

@MODULES.register_module()
class MultiGenerator(Encoder_Multi_Decoder):

    def encode_decode(self, img, img_metas, with_auxiliary_head=False):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, True)
        out = self._decode_head_forward_test(x, img_metas)
        for i in range(len(out)):
            if isinstance(out[i], list):
                for j in range(len(out[i])):
                    out[i][j] = resize(
                        input=out[i][j],
                        size=img.shape[2:],
                        mode='bilinear')
            else:
                out[i] = resize(
                    input=out[i],
                    size=img.shape[2:],
                    mode='bilinear')
        if with_auxiliary_head:
            auxilary_out = self._auxiliary_head_forward_test(x, img_metas)
            for i in range(len(auxilary_out)):
                auxilary_out[i] = resize(
                    input=auxilary_out[i],
                    size=img.shape[2:],
                    mode='bilinear')
            for i in range(len(out)):
                if isinstance(out[i], list):
                    out[i][0] = torch.cat((out[i][0], auxilary_out[i]))
                else:
                    out[i] = torch.cat((out[i], auxilary_out[i]))
        return out

    def _auxiliary_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        if isinstance(self.auxiliary_head, nn.ModuleList):
            seg_logits = []
            for idx, dec_head in enumerate(self.auxiliary_head):
                if len(x) == 2 and isinstance(x[0], list):
                    seg_logit = dec_head.forward_test(x[idx], img_metas, self.test_cfg)
                else:
                    seg_logit = dec_head.forward_test(x, img_metas, self.test_cfg)
                seg_logits.append(seg_logit)
        else:
            seg_logits = self.auxiliary_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_meta, with_auxiliary_head=False, **kwargs):
        seg_logits = self.encode_decode(img, img_meta, with_auxiliary_head=with_auxiliary_head)
        output = []
        for seg in seg_logits:
            if isinstance(seg, list):
                output.append(torch.sigmoid(seg[0]))
            else:
                output.append(torch.sigmoid(seg))
        return seg_logits, output


@MODULES.register_module()
class MultiTaskGenerator(Encoder_Decoder_Multi_task_group):

    def encode_decode(self, img, img_metas, with_auxiliary_head=False, **kwargs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, True)
        groups = self.decode_head.groups
        num_out = len(x)
        out = []
        for i in range(num_out):
            N, C, H, W = x[i].shape
            x[i] = x[i].repeat((1, groups, 1, 1))
            out.append(x[i])
        x = out
        out = self._decode_head_forward_test(x, img_metas, **kwargs)
        for i in range(len(out)):
            if isinstance(out[i], list):
                for j in range(len(out[i])):
                    out[i][j] = resize(
                        input=out[i][j],
                        size=img.shape[2:],
                        mode='bilinear')
            else:
                out[i] = resize(
                    input=out[i],
                    size=img.shape[2:],
                    mode='bilinear')
        if with_auxiliary_head:
            auxilary_out = self._auxiliary_head_forward_test(x, img_metas)
            for i in range(len(auxilary_out)):
                auxilary_out[i] = resize(
                    input=auxilary_out[i],
                    size=img.shape[2:],
                    mode='bilinear')
            for i in range(len(out)):
                if isinstance(out[i], list):
                    out[i][0] = torch.cat((out[i][0], auxilary_out[i]))
                else:
                    out[i] = torch.cat((out[i], auxilary_out[i]))
        return out

    def _auxiliary_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        if isinstance(self.auxiliary_head, nn.ModuleList):
            seg_logits = []
            for idx, dec_head in enumerate(self.auxiliary_head):
                if len(x) == 2 and isinstance(x[0], list):
                    seg_logit = dec_head.forward_test(x[idx], img_metas, self.test_cfg)
                else:
                    seg_logit = dec_head.forward_test(x, img_metas, self.test_cfg)
                seg_logits.append(seg_logit)
        else:
            seg_logits = self.auxiliary_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_meta, with_auxiliary_head=False, **kwargs):
        seg_logits = self.encode_decode(img, img_meta, with_auxiliary_head=with_auxiliary_head, **kwargs)
        output = []
        for seg in seg_logits:
            if isinstance(seg, list):
                output.append(torch.sigmoid(seg[0]))
            else:
                output.append(torch.sigmoid(seg))
        return seg_logits, output


@MODULES.register_module()
class Discriminator(nn.Module):

    def __init__(self,
                 in_channels,
                 **kwargs):
        super().__init__()
        # in_channels = 2
        base_channels = 64
        kernel_size = 4
        stride = (2, 2)
        padding = 1
        norm_cfg = None
        act_cfg = dict(type='LeakyReLU', negative_slope=0.2)
        self.conv1 = ConvModule(
            in_channels,
            base_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        self.conv2 = ConvModule(
            base_channels,
            2 * base_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        self.conv3 = ConvModule(
            2 * base_channels,
            4 * base_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        self.conv4 = ConvModule(
            4 * base_channels,
            8 * base_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        self.conv5 = ConvModule(
            8 * base_channels,
            1,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            norm_cfg=norm_cfg,
            act_cfg=None,
            **kwargs)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map.
            curr_scale (int): Current scale for discriminator. If in testing,
                you need to set it to the last scale.

        Returns:
            Tensor: Discriminative results.
        """
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        # x = self.up_sample(conv5)
        # x = self.sigmoid(x)
        return conv5
