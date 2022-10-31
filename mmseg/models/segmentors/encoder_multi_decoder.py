# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import os
import cv2


@SEGMENTORS.register_module()
class Encoder_Multi_Decoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Encoder_Multi_Decoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        if isinstance(decode_head, list):
            self.decode_head = nn.ModuleList()
            for head_cfg in decode_head:
                self.decode_head.append(builder.build_head(head_cfg))
        else:
            self.decode_head = builder.build_head(decode_head)
        # self.decode_head = builder.build_head(decode_head)
        # self.align_corners = self.decode_head.align_corners
        # self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, test_mode=False):
        """Extract features from images."""
        x = self.backbone(img, test_mode)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, True)
        out = self._decode_head_forward_test(x, img_metas)
        for i in range(len(out)):
            out[i] = resize(
                input=out[i],
                size=img.shape[2:],
                mode='bilinear')
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        num_task = len(self.decode_head)
        num_img = gt_semantic_seg.shape[0]
        batch_size_per_task = num_img // num_task
        for i in range(num_task):  # number of task
            idx = batch_size_per_task * i
            per_x = []
            for k in x:
                per_x.append(k[idx:(idx + batch_size_per_task)])
            per_img_metas = img_metas[idx:(idx + batch_size_per_task)]
            per_gt_semantic_seg = gt_semantic_seg[idx:(idx + batch_size_per_task)]
            loss_decode = self.decode_head[i].forward_train(per_x, per_img_metas,
                                                            per_gt_semantic_seg,
                                                            self.train_cfg)

            losses.update(add_prefix(loss_decode, f'decode_{i + 1}'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        if isinstance(self.decode_head, nn.ModuleList):
            seg_logits = []
            for idx, dec_head in enumerate(self.decode_head):
                if len(x) == 2 and isinstance(x[0], list):
                    seg_logit = dec_head.forward_test(x[idx], img_metas, self.test_cfg)
                else:
                    seg_logit = dec_head.forward_test(x, img_metas, self.test_cfg)
                seg_logits.append(seg_logit)
        else:
            seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        num_task = len(self.decode_head)
        num_img = gt_semantic_seg.shape[0]
        batch_size_per_task = num_img // num_task
        for i in range(num_task):  # number of task
            idx = batch_size_per_task * i
            per_x = []
            for k in x:
                per_x.append(k[idx:(idx + batch_size_per_task)])
            per_img_metas = img_metas[idx:(idx + batch_size_per_task)]
            per_gt_semantic_seg = gt_semantic_seg[idx:(idx + batch_size_per_task)]
            loss_decode = self.auxiliary_head[i].forward_train(per_x, per_img_metas,
                                                               per_gt_semantic_seg,
                                                               self.train_cfg)

            losses.update(add_prefix(loss_decode, f'aux_{i + 1}'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    @staticmethod
    def pad(batch, pad_dims, padding_value):
        ndim = batch[0].dim()
        assert ndim > pad_dims
        max_shape = [0 for _ in range(pad_dims)]
        for dim in range(1, pad_dims + 1):
            max_shape[dim - 1] = batch[0].size(-dim)
        for sample in batch:
            for dim in range(0, ndim - pad_dims):
                assert batch[0].size(dim) == sample.size(dim)
            for dim in range(1, pad_dims + 1):
                max_shape[dim - 1] = max(max_shape[dim - 1],
                                         sample.size(-dim))
        padded_samples = []
        for sample in batch:
            pad = [0 for _ in range(pad_dims * 2)]
            for dim in range(1, pad_dims + 1):
                pad[2 * dim -
                    1] = max_shape[dim - 1] - sample.size(-dim)
            padded_samples.append(
                F.pad(
                    sample, pad, value=padding_value))
        return padded_samples

    def train_step(self, data_batch, optimizer, **kwargs):
        if isinstance(data_batch, list):
            keys = ['img', 'img_metas', 'gt_semantic_seg']
            padded_batch = {}
            for key in keys:
                batch = []
                for i in range(len(data_batch)):
                    if key == 'img_metas':
                        batch.extend(data_batch[i][key])
                    else:
                        batch.append(data_batch[i][key])
                if key != 'img_metas':
                    batch = self.pad(batch, 2, 0)
                    padded_batch[key] = torch.cat(batch)
                else:
                    padded_batch[key] = batch
            data_batch = padded_batch
        # print(data_batch['img'].shape)
        # img1 = data_batch['gt_semantic_seg'][0].cpu().numpy().squeeze()
        # cv2.imwrite(os.path.join('liver', data_batch['img_metas'][0]['ori_filename'].split('.')[0]+'.png'), img1)
        # img2 = data_batch['gt_semantic_seg'][1].cpu().numpy().squeeze()
        # cv2.imwrite(os.path.join('kidney', data_batch['img_metas'][1]['ori_filename'].split('.')[0]+'.png'), img2)
        # img3 = data_batch['gt_semantic_seg'][2].cpu().numpy().squeeze()
        # cv2.imwrite(os.path.join('spleen', data_batch['img_metas'][2]['ori_filename'].split('.')[0]+'.png'), img3)

        # img1 = data_batch['img'][0].cpu().numpy().transpose(1, 2, 0)
        # cv2.imwrite(os.path.join('liver_img', data_batch['img_metas'][0]['ori_filename'].split('.')[0]+'.png'), img1)
        # img2 = data_batch['img'][1].cpu().numpy().transpose(1, 2, 0)
        # cv2.imwrite(os.path.join('kidney_img', data_batch['img_metas'][1]['ori_filename'].split('.')[0]+'.png'), img2)
        # img3 = data_batch['img'][2].cpu().numpy().transpose(1, 2, 0)
        # cv2.imwrite(os.path.join('spleen_img', data_batch['img_metas'][2]['ori_filename'].split('.')[0]+'.png'), img3)
        # img, gt-seg need to pad
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        # num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        # preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        preds = []
        for dec in self.decode_head:
            preds.append(img.new_zeros((batch_size, dec.num_classes, h_img, w_img)))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                for i in range(len(crop_seg_logit)):
                    preds[i] += F.pad(crop_seg_logit[i],
                               (int(x1), int(preds[i].shape[3] - x2), int(y1),
                                int(preds[i].shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        # preds = preds / count_mat
        for i in range(len(preds)):
            preds[i] = preds[i] / count_mat
        if rescale:
            for i in range(len(preds)):
                preds[i] = resize(
                    preds[i],
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        output = [[], []]
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            for i in range(len(seg_logit)):
                if isinstance(seg_logit[i], list):
                    seg_logit[i] = seg_logit[i][0]
                seg_logit[i] = resize(
                    seg_logit[i],
                    size=size,
                    mode='bilinear',
                    warning=False)

                # if i == 0:
                #     t = resize(
                #         seg_logit[i],
                #         size=size,
                #         mode='bilinear',
                #         warning=False)
                #     output[0].append(t)
                # else:
                #     t0 = resize(
                #         seg_logit[i][1],
                #         size=size,
                #         mode='bilinear',
                #         warning=False)
                #     output[0].append(t0)
                #     output[0].append((t + t0) / 2)
                #     t1 = resize(
                #         seg_logit[i][0],
                #         size=size,
                #         mode='bilinear',
                #         warning=False)
                #     output[1].append(t1)

        return seg_logit
        # return output

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
        output = []
        for i, seg in enumerate(seg_logit):
            if isinstance(seg, list):
                tmp = []
                for sseg in seg:
                    tmp.append(torch.sigmoid(sseg))
                output.append(tmp)
            else:
                output.append(torch.sigmoid(seg))

        # flip = img_meta[0]['flip']
        # if flip:
        #     flip_direction = img_meta[0]['flip_direction']
        #     assert flip_direction in ['horizontal', 'vertical']
        #     if flip_direction == 'horizontal':
        #         for i in range(len(output)):
        #             output[i] = output[i].flip(dims=(3, ))
        #     elif flip_direction == 'vertical':
        #         for i in range(len(output)):
        #             output[i] = output[i].flip(dims=(3, ))

        return output

    def simple_test(self, img, img_meta, rescale=True, test_mode=False, **kwargs):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        # todo
        output = [[], []]
        for i, seg in enumerate(seg_logit):
            if isinstance(seg, list):
                tmp = []
                for sseg in seg:
                    if sseg.shape[1] != 1:
                        seg_pred = (sseg > 0.5).int()
                    else:
                        seg_pred = sseg
                    seg_pred = seg_pred.squeeze(0).cpu().numpy()
                    tmp.append(seg_pred)
                seg_logit[i] = tmp
            else:
                if seg.shape[1] != 1 and seg.shape[1] != 4:
                    seg_pred = (seg > 0.5).int()
                else:
                    seg_pred = seg
                # seg_pred = (seg > 0.5).int()
                seg_pred = seg_pred.squeeze(0).cpu().numpy()
                seg_logit[i] = seg_pred
        if test_mode:
            seg_logit = [seg_logit, img_meta[0]['ori_filename'].split('.')[0]]
        return [seg_logit]

    def aug_test(self, imgs, img_metas, rescale=True, test_mode=False, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            for j in range(len(seg_logit)):
                seg_logit[j] += cur_seg_logit[j]
        for i in range(len(seg_logit)):
            seg_logit[i] /= len(imgs)
            seg_logit[i] = seg_logit[i].squeeze(0).cpu().numpy()
        if test_mode:
            seg_logit = [seg_logit, img_metas[0][0]['ori_filename'].split('.')[0]]
        # seg_pred = seg_logit.argmax(dim=1)
        # seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        # seg_pred = list(seg_pred)
        return [seg_logit]
