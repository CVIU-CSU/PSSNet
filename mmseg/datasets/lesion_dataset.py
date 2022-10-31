import os.path as osp
import warnings
from functools import reduce

import numpy as np
import torch
from mmcv.utils import print_log

from mmseg.core import lesion_metrics
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose, LoadAnnotations


@DATASETS.register_module()
class LesionDataset(CustomDataset):

    # CLASSES = ['bg', 'EX', 'HE', 'SE', 'MA', 'IRMA', 'NV']
    CLASSES = ['bg', 'EX', 'HE', 'SE', 'MA']
    # CLASSES = ['bg', 'IRMA']
    PALETTE = [
        [0, 0, 0],
        [128, 0, 0],  # EX: red
        [0, 128, 0],  # HE: green
        [128, 128, 0],  # SE: yellow
        [0, 0, 128],  # MA: blue
        # [128, 0, 128],  #IRMA: purple
        # [0, 128, 128],  #NV
    ]

    def __init__(self,
                pipeline,
                img_dir,
                img_suffix='.jpg',
                ann_dir=None,
                seg_map_suffix='.png',
                split=None,
                data_root=None,
                test_mode=False,
                ignore_index=255,
                reduce_zero_label=False,
                classes=None,
                palette=None,
                gt_seg_map_loader_cfg=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if isinstance(self.ann_dir, list):
                for i in range(len(self.ann_dir)):
                    if not (self.ann_dir[i] is None or osp.isabs(self.ann_dir[i])):
                        self.ann_dir[i] = osp.join(self.data_root, self.ann_dir[i])
            else:
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def evaluate(self, results, metric='mIoU', evaluate_per_image=False,logger=None, **kwargs):
        # return super(LesionDataset, self).evaluate(results, metric, logger, **kwargs)
        return self._evaluate(results, metric, evaluate_per_image, logger, **kwargs)

    def _evaluate(self, results, metric='mIoU', evaluate_per_image=False, logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        if not evaluate_per_image:
            ret_metrics = lesion_metrics(
                results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)  # evaluate
        else:
            ret_metrics = lesion_metrics_per_image(
                results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)  # evaluate
        iou, dice, ppv, s, aupr, auc_roc, mae = ret_metrics['IoU'], ret_metrics['dice'], ret_metrics['sp'], ret_metrics['se'], ret_metrics['AUPR'], ret_metrics['auc_roc'], ret_metrics['mae']
        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Dice', 'PPV', 'S', 'AUPR', 'auc_roc', 'MAE')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for i in range(num_classes):
            ppv_str = '{:.3f}'.format(ppv[i] * 100)
            s_str = '{:.3f}'.format(s[i] * 100)
            dice_str = '{:.3f}'.format(dice[i] * 100)
            iou_str = '{:.3f}'.format(iou[i] * 100)
            aupr_str = '{:.3f}'.format(aupr[i] * 100)
            auc_str = '{:.3f}'.format(auc_roc[i] * 100)
            mae_str = '{:.4f}'.format(mae[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, dice_str, ppv_str, s_str, aupr_str, auc_str, mae_str)

        mIoU = np.nanmean(np.nan_to_num(iou[-4:], nan=0))
        mDice = np.nanmean(np.nan_to_num(dice[-4:], nan=0))
        mPPV = np.nanmean(np.nan_to_num(ppv[-4:], nan=0))
        mS = np.nanmean(np.nan_to_num(s[-4:], nan=0))
        mAUPR = np.nanmean(np.nan_to_num(aupr[-4:], nan=0))
        mAUC = np.nanmean(np.nan_to_num(auc_roc[-4:], nan=0))
        mMAE = np.nanmean(np.nan_to_num(mae[-4:], nan=0))

        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mDice', 'mPPV', 'mS', 'mAUPR', 'mAUC', 'mMAE')

        iou_str = '{:.3f}'.format(mIoU * 100)
        dice_str = '{:.3f}'.format(mDice * 100)
        ppv_str = '{:.3f}'.format(mPPV * 100)
        s_str = '{:.3f}'.format(mS * 100)
        aupr_str = '{:.3f}'.format(mAUPR * 100)
        auc_str = '{:.3f}'.format(mAUC * 100)
        mae_str = '{:.4f}'.format(mMAE * 100)
        summary_str += line_format.format('global', iou_str, dice_str, ppv_str, s_str, aupr_str, auc_str, mae_str)
        print_log(summary_str, logger)

        # for save
        save_line = ''
        # IoU
        save_line += 'IoU:\n'
        for i in range(1, num_classes):
            save_line += '{:.2f}'.format(iou[i] * 100)
            save_line += ' '
        save_line += iou_str
        save_line += '\n'
        # Dice
        save_line += 'Dice:\n'
        for i in range(1, num_classes):
            save_line += '{:.2f}'.format(dice[i] * 100)
            save_line += ' '
        save_line += dice_str
        save_line += '\n'
        # AUPR
        save_line += 'AUPR:\n'
        for i in range(1, num_classes):
            save_line += '{:.2f}'.format(aupr[i] * 100)
            save_line += ' '
        save_line += aupr_str
        save_line += '\n'
        # AUC
        save_line += 'AUC:\n'
        for i in range(1, num_classes):
            save_line += '{:.2f}'.format(auc_roc[i] * 100)
            save_line += ' '
        save_line += auc_str
        save_line += '\n'
        # Se
        save_line += 'SE:\n'
        for i in range(1, num_classes):
            save_line += '{:.2f}'.format(s[i] * 100)
            save_line += ' '
        save_line += s_str
        save_line += '\n'
        # Sp
        save_line += 'SP:\n'
        for i in range(1, num_classes):
            save_line += '{:.2f}'.format(ppv[i] * 100)
            save_line += ' '
        save_line += ppv_str
        save_line += '\n'
        print_log(save_line, logger)


        eval_results['mIoU'] = mIoU
        eval_results['mDice'] = mDice
        eval_results['mPPV'] = mPPV
        eval_results['mS'] = mS
        eval_results['mAUPR'] = mAUPR
        eval_results['mAUC'] = mAUC
        eval_results['mMAE'] = mMAE

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return eval_results
