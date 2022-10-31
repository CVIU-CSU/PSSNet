from mmseg.core import add_prefix
from ..builder import SEGMENTORS
from .encoder_multi_decoder import Encoder_Multi_Decoder


@SEGMENTORS.register_module()
class Encoder_Multi_Decoder_Semi(Encoder_Multi_Decoder):

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        num_task = len(self.decode_head)
        num_img = gt_semantic_seg.shape[0] // 2
        batch_size_per_task = num_img // num_task
        for i in range(num_task):  # number of task, supervised loss
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

        for i in range(num_task):  # number of task, unsupervised loss
            idx = batch_size_per_task * i
            per_task_start = idx + num_img
            per_task_end = per_task_start + batch_size_per_task
            per_x = []
            for k in x:
                per_x.append(k[per_task_start:per_task_end])
            per_img_metas = img_metas[per_task_start:per_task_end]
            per_gt_semantic_seg = gt_semantic_seg[per_task_start:per_task_end]
            loss_decode = self.decode_head[i].forward_train(per_x, per_img_metas,
                                                            per_gt_semantic_seg,
                                                            self.train_cfg)
            for name, value in loss_decode.items():
                loss_decode[name] = 0.75 * value

            losses.update(add_prefix(loss_decode, f'decode_{i + 1}_semi'))
        return losses

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        num_task = 2
        num_img = gt_semantic_seg.shape[0] // 2
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

        for i in range(num_task):  # number of task
            idx = batch_size_per_task * i
            per_task_start = idx + num_img
            per_task_end = per_task_start + batch_size_per_task
            per_x = []
            for k in x:
                per_x.append(k[idx:(idx + batch_size_per_task)])
            per_img_metas = img_metas[idx:(idx + batch_size_per_task)]
            per_gt_semantic_seg = gt_semantic_seg[idx:(idx + batch_size_per_task)]
            loss_decode = self.auxiliary_head[i].forward_train(per_x, per_img_metas,
                                                               per_gt_semantic_seg,
                                                               self.train_cfg)
            for name, value in loss_decode.items():
                loss_decode[name] = 0.75 * value

            losses.update(add_prefix(loss_decode, f'aux_{i + 1}_semi'))
        return losses
