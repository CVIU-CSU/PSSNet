# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import mmcv
import cv2

import numpy as np
import torch


def sigmoid_intersect_and_union(pred_label, raw_label, num_classes):
    area_intersect = np.zeros((num_classes,), dtype=np.float)
    area_pred_label = np.zeros((num_classes,), dtype=np.float)
    area_label = np.zeros((num_classes,), dtype=np.float)

    for i in range(1, num_classes):
        pred = pred_label[i - 1] > 0.5
        if i == 1:
            label = raw_label > 0
        else:
            label = raw_label == i
        area_intersect[i] = np.sum(label & pred)
        area_pred_label[i] = np.sum(pred)
        area_label[i] = np.sum(label)

    # pred = np.zeros(pred_label[0].shape, dtype=np.int)
    # label = np.zeros(raw_label.shape, dtype=np.int)
    # for i in range(1, num_classes - 1):
    #     pred_i = pred_label[i - 1]
    #     pred[pred_i == 1] = 1
    # label[raw_label > 0] = 1
    # ai = np.sum(label & pred)
    # area_intersect[-1] = ai
    # area_pred_label[-1] = area_pred_label[1] + area_pred_label[2]
    # area_label[-1] = area_label[1] + area_label[2]

    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def sigmoid_metrics(results, gt_seg_maps, num_classes):
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        if isinstance(result, list):
            for res in result:
                if res.shape[0] == 2:
                    result = res
        area_intersect, area_union, area_pred_label, area_label = \
            sigmoid_intersect_and_union(
                result, gt_seg_map, num_classes)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
           total_area_label


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    # multi task
    if isinstance(pred_label, list):
        pred_label = pred_label[0]
    # End
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    pred_label[pred_label > 0] = 1
    label[label > 0] = 1
    inter = pred_label[pred_label == label]
    ai = torch.histc(
        inter.float(), bins=(2), min=0, max=1)
    area_intersect[-1] = ai[1]
    area_pred_label[-1] = area_pred_label[1] + area_pred_label[2]
    area_label[-1] = area_label[1] + area_label[2]

    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes,), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
           total_area_label


def eval_ODOC_metrics(results,
                      gt_seg_maps,
                      num_classes,
                      ignore_index,
                      metrics=['mIoU'],
                      nan_to_num=None,
                      label_map=dict(),
                      reduce_zero_label=False,
                      beta=1,
                      use_sigmoid=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(results[0], list):
        use_sigmoid = len(results[0][0].shape) != 2
    else:
        use_sigmoid = len(results[0].shape) != 2

    if not use_sigmoid:
        total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    else:
        total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = sigmoid_metrics(
            results, gt_seg_maps, num_classes)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta, use_sigmoid)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1,
                          use_sigmoid=False):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    ret_metrics = OrderedDict()

    iou = total_area_intersect / total_area_union
    dice = 2 * total_area_intersect / (
            total_area_pred_label + total_area_label)
    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label

    ret_metrics['IoU'] = iou
    ret_metrics['Dice'] = dice
    ret_metrics['Precision'] = precision
    ret_metrics['Recall'] = recall

    if not use_sigmoid:
        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
