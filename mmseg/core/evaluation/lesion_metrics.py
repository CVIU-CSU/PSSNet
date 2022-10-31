# NEW

SIGMOID_THRESH = 0.5

import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from collections import OrderedDict
# import matplotlib.pyplot as plt

np.seterr(invalid='ignore')

"""
results:
gt:
[ (2848, 4288),  ...]

results {0, 1, 2, 3 ,4}
gt {0, 1, 2, 3, 4}

0 stands for background
"""


# softmax
def softmax_confused_matrix(pred_label, label, num_classes):
    tp = pred_label[pred_label == label]

    area_p, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_tp, _ = np.histogram(tp, bins=np.arange(num_classes + 1))
    area_gt, _ = np.histogram(label, bins=np.arange(num_classes + 1))

    area_fn = area_gt - area_tp

    return area_p, area_tp, area_fn


def softmax_metrics(results, gt_seg_maps, num_classes):
    """
    :param results:  {0, 1, 2, 3 ,4}
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs

    total_p = np.zeros((num_classes,), dtype=np.float)
    total_tp = np.zeros((num_classes,), dtype=np.float)
    total_fn = np.zeros((num_classes,), dtype=np.float)
    maupr = np.zeros((num_classes,), dtype=np.float)

    for i in range(num_imgs):
        if isinstance(results[i], list):
            result = results[i][1]    # multi task
        else:
            result = results[i]

        p, tp, fn = softmax_confused_matrix(result, gt_seg_maps[i], num_classes)
        total_p += p
        total_tp += tp
        total_fn += fn

    return total_p, total_tp, total_fn, maupr


# sigmoid
def sigmoid_confused_matrix(pred_logit, raw_label, num_classes, thresh, use_sigmoid):
    # assert pred_logit.shape[0] == num_classes - 1

    class_p = np.zeros((num_classes,), dtype=np.float)
    class_tp = np.zeros((num_classes,), dtype=np.float)
    class_fn = np.zeros((num_classes,), dtype=np.float)
    class_mae = np.zeros((num_classes,), dtype=np.float)

    for i in range(1, num_classes):
        if use_sigmoid:
            pred = pred_logit[i - 1] > thresh
        else:
            pred = pred_logit[i] > thresh
        label = raw_label == i
        class_tp[i] = np.sum(label & pred)
        class_p[i] = np.sum(pred)
        class_fn[i] = np.sum(label) - class_tp[i]
        class_mae[i] = np.mean(np.abs(pred ^ label))

    return class_p, class_tp, class_fn, class_mae


# sigmoid
def sigmoid_metrics(results, gt_seg_maps, num_classes, compute_aupr, use_sigmoid):
    num_imgs = len(results)

    if compute_aupr:
        # threshs = np.linspace(0, 1, 33)  # 0.03125
        threshs = np.linspace(0, 1, 11)  # 0.1
    else:
        threshs = [SIGMOID_THRESH]

    total_p_list = []
    total_tp_list = []
    total_fn_list = []
    total_mae_list = []

    for thresh in threshs:
        total_p = np.zeros((num_classes,), dtype=np.float)
        total_tp = np.zeros((num_classes,), dtype=np.float)
        total_fn = np.zeros((num_classes,), dtype=np.float)
        total_mae = np.zeros((num_classes,), dtype=np.float)

        for i in range(num_imgs):
            if isinstance(results[i], tuple):
                result = results[i][0]
            elif isinstance(results[i], list):
                for res in results[i]:
                    if use_sigmoid:
                        if res.shape[0] == 4:
                            result = res
                            break
                    else:
                        if res.shape[0] == 5:
                            result = res
                            break
            else:
                result = results[i]

            p, tp, fn, mae = sigmoid_confused_matrix(result, gt_seg_maps[i], num_classes, thresh, use_sigmoid)
            total_p += p
            total_tp += tp
            total_fn += fn
            total_mae += mae

        total_p_list.append(total_p)
        total_tp_list.append(total_tp)
        total_fn_list.append(total_fn)
        total_mae_list.append(total_mae)

    if len(threshs) > 1:
        index = int(np.argmax(threshs == 0.5))
    else:
        index = 0
    total_p = total_p_list[index]
    total_tp = total_tp_list[index]
    total_fn = total_fn_list[index]
    total_mae = total_mae_list[index]

    maupr = np.zeros((num_classes,), dtype=np.float)
    total_p_list = np.stack(total_p_list)   
    total_tp_list = np.stack(total_tp_list)
    total_fn_list = np.stack(total_fn_list)

    ppv_list = np.nan_to_num(total_tp_list / total_p_list, nan=1)
    s_list = np.nan_to_num(total_tp_list / (total_tp_list + total_fn_list), nan=0)

    if compute_aupr:
        for i in range(1, len(maupr)):
            x = s_list[:, i]
            y = ppv_list[:, i]
            maupr[i] = auc(x, y)

    return total_p, total_tp, total_fn, maupr, total_mae / num_imgs


def roc(results, gt_seg_maps, num_classes, use_sigmoid):
    num_imgs = len(results)
    auc_roc = np.zeros((num_classes, ), dtype=float)
    # aupr = np.zeros((num_classes, ), dtype=float)
    num_lesion = [1] + [0] * (num_classes - 1)


    for result, gt_seg_map in zip(results, gt_seg_maps):
        if isinstance(result, tuple):
            result = result[0]
        elif isinstance(result, list):
            for res in result:
                if use_sigmoid:
                    if res.shape[0] == 4:
                        result = res
                        break
                else:
                    if res.shape[0] == 5:
                        result = res
                        break
        for c in range(1, num_classes):
            if use_sigmoid:
                pred = result[c-1].flatten()
            else:
                pred = result[c].flatten()
            gt = (gt_seg_map == c).flatten()

            if (gt == 0).all():
                pass
                # aupr[c] += 1.0
            else:
                fpr, tpr, _ = roc_curve(gt, pred, pos_label=1)
                # precision, recall, _ = precision_recall_curve(gt, pred, pos_label=1)
                # fig = plt.figure()
                # plt.plot(recall, precision)
                # plt.savefig(f'pr/{index}_{c}.png')
                # plt.close(fig)
                auc_roc[c] += auc(fpr, tpr)
                num_lesion[c] += 1
                # aupr[c] += auc(recall, precision)

    return auc_roc / num_lesion  #, aupr / num_imgs


def lesion_metrics(results, gt_seg_maps, num_classes, ignore_index=None, nan_to_num=None):
    """
    :param results: feature map after sigmoid of softmax
    """
    gt_seg_maps_tmp = []
    for gt in gt_seg_maps:
        gt_seg_maps_tmp.append(gt)
    gt_seg_maps = gt_seg_maps_tmp

    compute_aupr = True
    # use_sigmoid = True
    if isinstance(results[0], tuple):
        use_sigmoid = results[0][0].shape[0] == num_classes - 1
    elif isinstance(results[0], list):
        for res in results[0]:
            if res.shape[0] == num_classes - 1:
                use_sigmoid = True
                break
            elif res.shape[0] == num_classes:
                use_sigmoid = False
                break
    else:
        use_sigmoid = (results[0].shape[0] == num_classes - 1)

    auc_roc = roc(results, gt_seg_maps, num_classes, use_sigmoid)

    total_p, total_tp, total_fn, maupr, mae = sigmoid_metrics(
        results, gt_seg_maps, num_classes, compute_aupr, use_sigmoid)

    ppv = total_tp / total_p
    s = total_tp / (total_tp + total_fn)
    dice = 2 * total_tp / (total_p + total_tp + total_fn)
    # f1 = (s * ppv * 2) / (s + ppv)
    iou = total_tp / (total_p + total_fn)

    ret_metrics = OrderedDict()
    ret_metrics['IoU'] = iou
    ret_metrics['AUPR'] = maupr
    ret_metrics['se'] = s
    ret_metrics['sp'] = ppv
    ret_metrics['dice'] = dice
    ret_metrics['auc_roc'] = auc_roc
    ret_metrics['mae'] = mae

    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics


if __name__ == '__main__':
    # test code
    shape = [3, 4]
    num_classes = 4  # include background
    num = 2
    use_sigmoid = True
    aupr = False

    pred = [(np.random.random([num_classes, shape[0], shape[1]]), use_sigmoid, aupr) for i in range(num)]
    label = [np.random.randint(0, num_classes + 1, shape) for i in range(num)]

    res = lesion_metrics(pred, label, num_classes + 1)
    for i in res:
        print(i)
