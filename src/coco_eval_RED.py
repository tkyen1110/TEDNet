# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Utility to compute coco metrics with a time tolerance
"""
import io
from contextlib import redirect_stdout

import numpy as np

from numba import jit
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator(object):
    """Wrapper Class for coco metrics.

    It is equivalent to the `evaluate_detection` function but will use less memory when used for a large number of
    files. In this case list of boxes can be fed in several go before being accumulated.

    Args:
        classes (tuple): all class names
        height: frame height, used to determine if the box is considered big medium or small
        width: frame width, used to determine if the box is considered big medium or small
        time_tol (float): half range time tolerance to match all_ts ('t' are matched with +/- time_tol) (in us)
        eval_rate (int): if eval rate > 0 we evaluate every *eval_rate* (us), the windows are centered around
            [0, eval_rate, 2* eval_rate , ... etc]
            otherwise, the windows are centered round every timestamp with at least one box (gt or detection)
        verbose (boolean): if True, print the COCO APIs prints.

    Returns:
        coco_kpi (dict): all kpi results

    Examples:
        >>> coco_wrapper = CocoEvaluator()
        >>> coco_wrapper.partial_eval([gt_bbox1], [dt_bbox_2])
        >>> coco_wrapper.partial_eval(gt_box_list, dt_box_list)
        >>> result_dict = coco_wrapper.accumulate()
    """

    def __init__(self, classes=("car", "pedestrian"), height=240, width=304, time_tol=40000, eval_rate=-1,
                 verbose=False):
        self._num_images = 0
        self._num_annotations = 0
        self._num_dets = 0
        self._coco_res = []
        if isinstance(classes, str):
            classes = (classes,)
        self.classes = classes
        self.height = height
        self.width = width
        self.time_tol = time_tol
        self.eval_rate = eval_rate
        self.verbose = verbose

    def partial_eval(self, gt_boxes_list, dt_boxes_list):
        """Compute partial results for KPIs given a list of bounding box ground truth vectors and the list of matching
        predictions.

        Note that timestamps 't' must be increasing inside a given vector.
        """
        flattened_gt, flattened_dt = _flatten_boxes_list(gt_boxes_list, dt_boxes_list,
                                                         time_tol=self.time_tol, eval_rate=self.eval_rate)

        console_output = io.StringIO()
        with redirect_stdout(console_output):
            partial_res, self._num_images, self._num_annotations, self._num_dets = _coco_eval_partial(
                flattened_gt, flattened_dt, self.height, self.width,
                num_images=self._num_images, num_annotations=self._num_annotations, num_dets=self._num_dets,
                labelmap=self.classes)

        if self.verbose:
            print(console_output.getvalue())

        # if there is already a coco_eval object we can discard it but keep only the partial results.
        if self._coco_res:
            self._coco_res[-1] = self._coco_res[-1].evalImgs
        self._coco_res.append(partial_res)

    def accumulate(self):
        """Accumulates all previously compared detections and ground truth into a single set of COCO KPIs.

        Returns:
            eval_dict (dict): dict with keys KPI names and float values KPIs.
        """
        if not self._coco_res:
            print("Warning : No boxes were added to the evaluation !")
            return {'mean_ap75': -1.0, 'mean_ar_medium': -1.0, 'mean_ap50': -1.0, 'mean_ap_big': -1.0,
                    'mean_ap': -1.0, 'mean_ap_small': -1.0, 'mean_ap_medium': -1.0, 'mean_ar_big': -1.0,
                    'mean_ar_small': -1.0, 'mean_ar': -1.0}
        coco_eval = self._coco_res[-1]
        # concatenate the list of partial results.
        for coco_evalImgs in reversed(self._coco_res[:-1]):
            num_cats = len(coco_eval.params.catIds)
            num_ranges = len(coco_eval.params.areaRng)
            coco_eval.evalImgs = _merge_eval_list(coco_evalImgs, coco_eval.evalImgs, num_cats, num_ranges)

        # all image ids should be considered during accumulation
        coco_eval.params.imgIds = np.arange(1, self._num_images + 1, dtype=int)
        coco_eval._paramsEval.imgIds = np.arange(1, self._num_images + 1, dtype=int)

        console_output = io.StringIO()
        with redirect_stdout(console_output):
            coco_eval.accumulate()
            stats = summarize(coco_eval, self.verbose)

        if self.verbose:
            print(console_output.getvalue())

        return {
            "mean_ap": stats[0], "mean_ap50": stats[1], "mean_ap75": stats[2],
            "mean_ap_small": stats[3], "mean_ap_medium": stats[4], "mean_ap_big": stats[5],
            "mean_ar": stats[8],
            "mean_ar_small": stats[9], "mean_ar_medium": stats[10], "mean_ar_big": stats[11]
        }


def evaluate_detection(gt_boxes_list, dt_boxes_list, classes=("car", "pedestrian"), height=240, width=304,
                       time_tol=40000, eval_rate=-1):
    """
    Evaluates detection kpis on gt & dt arrays, be advised ts should be strictly increasing
    if eval_rate =-1 the kpi is computed only at the timestamps where there is gt
    (if necessary fill the gt with a background box in frames without gt)

    Args:
        gt_boxes_list: merged list of all ground-truth boxes
        dt_boxes_list: merged list of all detection boxes
        classes: all class names
        height: frame height
        width: frame width
        time_tol (float): half range time tolerance to match all_ts ('t' are matched with +/- time_tol) (in us)
        eval_rate (int): if eval rate > 0 we evaluate every *eval_rate* (us), the windows are centered around
            [0, eval_rate, 2* eval_rate , ... etc]
            otherwise, the windows are centered round every timestamp with at least one box (gt or detection)

    Returns:
        coco_kpi (dict): all kpi results

    """
    if isinstance(classes, str):
        classes = (classes,)
    flattened_gt, flattened_dt = _flatten_boxes_list(gt_boxes_list, dt_boxes_list,
                                                     time_tol=time_tol, eval_rate=eval_rate)

    if not len(flattened_gt):
        return {'mean_ap75': -1.0, 'mean_ar_medium': -1.0, 'mean_ap50': -1.0, 'mean_ap_big': -1.0,
                'mean_ap': -1.0, 'mean_ap_small': -1.0, 'mean_ap_medium': -1.0, 'mean_ar_big': -1.0,
                'mean_ar_small': -1.0, 'mean_ar': -1.0}
    coco_eval, _, _, _ = _coco_eval_partial(flattened_gt, flattened_dt, height, width,
                                            num_images=0, num_annotations=0, num_dets=0, labelmap=classes)
    coco_eval.accumulate()
    stats = summarize(coco_eval)

    return {
        "mean_ap": stats[0], "mean_ap50": stats[1], "mean_ap75": stats[2],
        "mean_ap_small": stats[3], "mean_ap_medium": stats[4], "mean_ap_big": stats[5],
        "mean_ar": stats[8],
        "mean_ar_small": stats[9], "mean_ar_medium": stats[10], "mean_ar_big": stats[11]
    }


@jit
def match_times(all_ts, boxes_no_tol, boxes_tol, time_tol):
    """
    Matches ground truth boxes and ground truth detections at all timestamps using a specified tolerance
    returns a list of boxes vectors

    Args:
        all_ts: all timestamps of evaluation
        boxes_no_tol (np.ndarray): bounding boxes with 't' time field (those 't' must be a subset of all_ts)
        boxes_tol (np.ndarray): bounding boxes with 't' time field (those 't' are matched to all_ts using a 2 * time_tol interval)
        time_tol (float): half range time tolerance to match all_ts ('t' are matched with +/- time_tol) (in us)
    Returns:
        windowed_boxes_no_tol (list): list of np.ndarray computed from boxes_no_tol
        windowed_boxes_tol (list): list of np.ndarray computed from boxes_tol
    """
    box_no_tol_size = len(boxes_no_tol)
    boxes_tol_size = len(boxes_tol)

    windowed_boxes_no_tol = []
    windowed_boxes_tol = []

    low_idx_no_tol, high_idx_no_tol = 0, 0
    low_idx_tol, high_idx_tol = 0, 0
    for ts in all_ts:

        while low_idx_no_tol < box_no_tol_size and boxes_no_tol[low_idx_no_tol]['t'] < ts:
            low_idx_no_tol += 1
        # the high index is at least as big as the low one
        high_idx_no_tol = max(low_idx_no_tol, high_idx_no_tol)
        while high_idx_no_tol < box_no_tol_size and boxes_no_tol[high_idx_no_tol]['t'] <= ts:
            high_idx_no_tol += 1

        # detection are allowed to be inside a window around the right detection timestamp
        low = ts - time_tol
        high = ts + time_tol
        while low_idx_tol < boxes_tol_size and boxes_tol[low_idx_tol]["t"] < low:
            low_idx_tol += 1
        # the high index is at least as big as the low one
        high_idx_tol = max(low_idx_tol, high_idx_tol)
        while high_idx_tol < boxes_tol_size and boxes_tol[high_idx_tol]['t'] <= high:
            high_idx_tol += 1

        windowed_boxes_no_tol.append(boxes_no_tol[low_idx_no_tol:high_idx_no_tol])
        windowed_boxes_tol.append(boxes_tol[low_idx_tol:high_idx_tol])

    return windowed_boxes_no_tol, windowed_boxes_tol


def summarize(coco_eval, verbose=False):
    """
    Computes and displays summary metrics for evaluation results.
    Note this function can *only* be applied on the default parameter setting
    """
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
            if verbose:
                for i in range(0, len(p.catIds)):
                    s_c = np.mean(s[:, :, i, :])
                    if s_c != -1:
                        iStr_cat = '{:<18} {} of category {:>3d} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {' \
                                   ':0.3f} '
                        print(iStr_cat.format(titleStr, typeStr, i, iouStr, areaRng, maxDets, s_c))
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
            if verbose:
                for i in range(0, len(p.catIds)):
                    s_c = np.mean(s[:, i, :])
                    if s_c != -1:
                        iStr_cat = '{:<18} {} of category {:>3d} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {' \
                                   ':0.3f} '
                        print(iStr_cat.format(titleStr, typeStr, i, iouStr, areaRng, maxDets, s_c))
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        if verbose:
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s

    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=coco_eval.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=coco_eval.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=coco_eval.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        return stats

    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats
    if not coco_eval.eval:
        raise Exception('Please run accumulate() first')
    iouType = coco_eval.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    stats = summarize()
    return stats


def _flatten_boxes_list(gt_boxes_list, dt_boxes_list, time_tol=40000, eval_rate=-1):
    """put detections and gt boxes in parallel lists according to evaluation rate and time tolerance"""
    flattened_gt = []
    flattened_dt = []
    for gt_boxes, dt_boxes in zip(gt_boxes_list, dt_boxes_list):

        assert np.all(gt_boxes['t'][1:] >= gt_boxes['t'][:-1])
        assert np.all(dt_boxes['t'][1:] >= dt_boxes['t'][:-1])

        all_ts = np.unique(np.concatenate((gt_boxes['t'], dt_boxes['t'])))
        if eval_rate > 0 and len(all_ts):
            assert ((dt_boxes["t"] % eval_rate) == 0).all()
            last_ts = np.max(all_ts)
            all_ts = np.arange(eval_rate, last_ts, step=eval_rate)
            dt_win, gt_win = match_times(all_ts=all_ts, boxes_no_tol=dt_boxes, boxes_tol=gt_boxes, time_tol=time_tol)
        else:
            gt_win, dt_win = match_times(all_ts=all_ts, boxes_no_tol=gt_boxes, boxes_tol=dt_boxes, time_tol=time_tol)
        flattened_gt = flattened_gt + gt_win
        flattened_dt = flattened_dt + dt_win
    return flattened_gt, flattened_dt


def _merge_eval_list(list_a, list_b, num_cat, num_rng):
    res = []
    num_ids_a = len(list_a) // (num_cat * num_rng)
    num_ids_b = len(list_b) // (num_cat * num_rng)
    for c in range(num_cat):
        for r in range(num_rng):
            res.extend(list_a[:num_ids_a])
            res.extend(list_b[:num_ids_b])
            list_a = list_a[num_ids_a:]
            list_b = list_b[num_ids_b:]
    return res


def _coco_eval_partial(gts, detections, height, width, num_images=0, num_annotations=0, num_dets=0,
                       labelmap=("car", "pedestrian")):
    """
    Simple helper function wrapping around COCO's Python API

    Args:
        gts: iterable of numpy boxes for the ground truth
        detections: iterable of numpy boxes for the detections
        height (int): frame height
        width (int): frame width
        num_images (int): number of images previously evaluated (in case of multiple call to evaluate)
        num_annotations (int): number of annotations previously evaluated (in case of multiple call to evaluate)
        num_dets (int): number of detections previously evaluated (in case of multiple call to evaluate)
        labelmap (list): iterable of class labels
    """
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(labelmap)]

    dataset, results = _to_coco_format(gts, detections, categories, num_images=num_images,
                                       num_annotations=num_annotations, num_dets=num_dets, height=height, width=width)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()

    coco_pred = COCO()
    coco_pred.dataset = results
    if len(results["annotations"]):
        coco_pred.createIndex()

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(num_images + 1, num_images + len(gts) + 1, dtype=int)

    coco_eval.evaluate()

    num_images = num_images + len(gts)
    if len(dataset["annotations"]):
        num_annotations = dataset['annotations'][-1]['id']
    if len(results["annotations"]):
        num_dets = results['annotations'][-1]['id']
    return coco_eval, num_images, num_annotations, num_dets


def _to_coco_format(gts, detections, categories, num_images=0, num_annotations=0, num_dets=0, height=240, width=304):
    """Utility function producing our data in a COCO usable format

    Args:
        gts: ground-truth boxes
        detections: detection boxes
        categories: class for pycoco api
        num_images (int): number of images previously evaluated (in case of multiple call to evaluate)
        num_annotations (int): number of annotations previously evaluated (in case of multiple call to evaluate)
        height: frame height
        width: frame width
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = num_images + image_id + 1

        images.append({"id": im_id, "height": height, "width": width})

        for bbox in gt:
            x1, y1 = bbox['x'], bbox['y']
            w, h = bbox['w'], bbox['h']
            area = w * h

            annotation = {
                "area": float(area), "iscrowd": False,
                "image_id": im_id, "bbox": [x1, y1, w, h],
                "category_id": int(bbox['class_id']) + 1, "id": num_annotations + len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox in pred:
            x1 = bbox['x']
            x2 = bbox['x'] + bbox['w']
            y1 = bbox['y']
            y2 = bbox['y'] + bbox['h']
            image_result = {
                'image_id': im_id,
                'category_id': int(bbox['class_id']) + 1, 'score': float(bbox['class_confidence']),
                'bbox': [x1, y1, bbox['w'], bbox['h']], 'segmentation': [[x1, y1, x1, y2, x2, y2, x2, y1]],
                'area': float(bbox['w'] * bbox['h']), 'id': 1 + len(results) + num_dets, 'iscrowd': 0
            }

            results.append(image_result)

    dataset = {"info": {}, "licenses": [], "type": 'instances',
               "images": images, "annotations": annotations, "categories": categories}
    result_dataset = {"info": {}, "licenses": [], "type": 'instances',
                      "images": images, "annotations": results, "categories": categories}
    return dataset, result_dataset
