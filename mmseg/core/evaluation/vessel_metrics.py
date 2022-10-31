# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import mmcv
import cv2

import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False,
                        use_sigmoid=False):
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

    if use_sigmoid:
        pred_label = (pred_label[0] > 0.5).int()

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
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False,
                              use_sigmoid=False):
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
    
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        if isinstance(result, list):
            for res in result:
                if res.shape[0] == 1:
                    result = res
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label, use_sigmoid)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label



def eval_auc(results, gt_seg_maps, use_sigmoid):
    num_imgs = len(results)
    aucs = 0
    #pred_list = []
    #gt_list = []
    for result, gt_seg_map in zip(results, gt_seg_maps):
        if isinstance(result, list):
            result = result[0]
        y = gt_seg_map.flatten()
        if not use_sigmoid:
            pred = result[1].flatten()
        else:
            pred = result.flatten()
        #pred_list += pred
        #gt_list += y
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        tmp = auc(fpr, tpr)
        aucs += tmp
    mauc = aucs / num_imgs
    return mauc


def precision_recall(results, gt_seg_maps, use_sigmoid):
    num_imgs = len(results)
    se = 0
    sp = 0
    ppv = 0

    for result, gt_seg_map in zip(results, gt_seg_maps):
        if isinstance(result, list):
            result = result[0]
        if not use_sigmoid:
            result = (result[1] * 255).astype(np.uint8)
        else:
            result = (result[0] * 255).astype(np.uint8)
        _, result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)
        result = result // 255
        gt_background = (gt_seg_map == 0).sum()
        gt_vessel = (gt_seg_map == 1).sum()
        pred_background = (result == 0).sum()
        pred_vessel = (result == 1).sum()
        tp = result[result==gt_seg_map]
        tp_background = (tp == 0).sum()
        tp_vessel = (tp == 1).sum()
        se += tp_vessel / gt_vessel
        sp += tp_background / gt_background
        ppv += tp_vessel / pred_vessel
    se /= num_imgs
    sp /= num_imgs
    ppv /= num_imgs
    return se, sp, ppv

def eval_vessel_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1,
                 use_sigmoid=True):
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

    gt_seg_maps_tmp = []
    for gt_seg_map in gt_seg_maps:
        gt_seg_maps_tmp.append(gt_seg_map)
    gt_seg_maps = gt_seg_maps_tmp
    auc = eval_auc(results, gt_seg_maps, use_sigmoid)
    se, sp, ppv = precision_recall(results, gt_seg_maps, use_sigmoid)

    if not use_sigmoid:
        for i in range(len(results)):
            results[i] = results[i].argmax(axis=0)


    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label, use_sigmoid)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    ret_metrics['auc'] = np.array([np.nan, auc])
    ret_metrics['se'] = np.array([np.nan, se])
    ret_metrics['sp'] = np.array([np.nan, sp])
    ret_metrics['ppv'] = np.array([np.nan, ppv])
    tpn, lpn = 0, 0
    for item in total_area_intersect:
        tpn += item
    for item in total_area_label:
        lpn += item
    acc = tpn / lpn
    ret_metrics['ACC'] = np.array([np.nan, acc])
    return ret_metrics



def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
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
