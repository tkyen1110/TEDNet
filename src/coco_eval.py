# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

# This file is modified from the original code at
# https://github.com/hamarh/HMNet_pth/blob/main/experiments/detection/scripts/coco_eval.py which is modified from
# https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/metrics/coco_eval.py
# The list of modifications are as follows:
# (1) (HMNet)  add revised function "_match_times_rev" that matches GT boxes to detected boxes with nearest timestamp
# (2) (HMNet)  "evaluate_detection" returns COCOeval instance
# (3) (TEDNet) "evaluate_detection_RED" uses the same evaluation code as RED model 

"""
Compute the COCO metric on bounding box files by matching timestamps

Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from coco_eval_RED import CocoEvaluator
try:
    from metavision_core.event_io.py_reader import EventDatReader
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    record = True
except ModuleNotFoundError as err:
    print(err)
    record = False

def nms(box_events, scores, iou_thresh=0.5):
    """NMS on box_events

    Args:
        box_events (np.ndarray): nx1 with dtype EventBbox, the sorting order of those box is used as a
            a criterion for the nms.
        scores (np.ndarray): nx1 dtype of plain dtype, needs to be argsortable.
        iou_thresh (float): if two boxes overlap with more than `iou_thresh` (intersection over union threshold)
            with each other, only the one with the highest criterion value is kept.

    Returns:
        keep (np.ndarray): Indices of the box to keep in the input array.
    """
    x1 = box_events['x']
    y1 = box_events['y']
    x2 = box_events['x'] + box_events['w']
    y2 = box_events['y'] + box_events['h']

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return sorted(keep)

def record_video(gt_win, dt_win, npy_file, dt_folder, event_folder, verbose=False):
    event_file = npy_file.replace("_bbox.npy", "_td.dat")
    event_file_path = os.path.join(event_folder, event_file)
    video_folder = os.path.join(dt_folder, "videos")
    os.makedirs(video_folder, exist_ok=True)
    video_file_path = os.path.join(video_folder, npy_file.replace("_bbox.npy", ".mp4"))

    event_dat = EventDatReader(event_file_path)
    ev_height, ev_width = event_dat.get_size()
    delta_t = 50000

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    frame_rate = 1
    video_writer = cv2.VideoWriter(video_file_path, fourcc, frame_rate, (ev_width*2, ev_height))
    colors = [[0,0,255], [0,255,0]]

    for gt_boxes, dt_boxes in zip(gt_win, dt_win):
        end_time = gt_boxes['t'][0]
        start_time = end_time - delta_t
        event_dat.seek_time(start_time)
        events = event_dat.load_delta_t(delta_t=delta_t)

        image_all = np.zeros((ev_height, ev_width*2, 3), dtype=np.uint8)
        BaseFrameGenerationAlgorithm.generate_frame(events, image_all[:,:ev_width, :])
        image_all[:,ev_width:, :] = image_all[:,:ev_width, :].copy()

        cv2.putText(image_all, '{} ms / {} ms'.format(start_time//1000, (end_time+1)//1000), 
            (ev_width-200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_all, 'GT',
                    (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_all, 'DT',
                    (ev_width+10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        image_all = cv2.line(image_all, (ev_width,0), (ev_width,ev_height), (0,255,255), 1)

        for gt_box in gt_boxes:
            t, x, y, w, h, class_id, track_id, confidence = gt_box
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image_all, (x, y), (x+w, y+h), colors[class_id], 1)

        for dt_box in dt_boxes:
            t, x, y, w, h, class_id, track_id, confidence = dt_box
            if confidence > 0.0:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(image_all, (ev_width+x, y), (ev_width+x+w, y+h), colors[class_id], 1)
        video_writer.write(image_all)
    video_writer.release()

def evaluate_detection(gt_boxes_list, dt_boxes_list, npy_file_list, dt_folder, event_folder, 
                       classes=("car", "pedestrian"), height=240, width=304, time_tol=50000):
    """
    Compute detection KPIs on list of boxes in the numpy format, using the COCO python API
    https://github.com/cocodataset/cocoapi
    KPIs are only computed on timestamps where there is actual at least one box
    (fully empty frames are not considered)

    :param gt_boxes_list: list of numpy array for GT boxes (one per file)
    :param dt_boxes_list: list of numpy array for detected boxes
    :param classes: iterable of classes names
    :param height: int for box size statistics
    :param width: int for box size statistics
    :param time_tol: int size of the temporal window in micro seconds to look for a detection around a gt box
    """
    flattened_gt = []
    flattened_dt = []
    for gt_boxes, dt_boxes, npy_file in zip(gt_boxes_list, dt_boxes_list, npy_file_list):

        assert np.all(gt_boxes['t'][1:] >= gt_boxes['t'][:-1])
        assert np.all(dt_boxes['t'][1:] >= dt_boxes['t'][:-1])

        all_ts = np.unique(gt_boxes['t'])
        n_steps = len(all_ts)

        gt_win, dt_win = _match_times_rev(all_ts, gt_boxes, dt_boxes, time_tol)
        if event_folder != None and record:
            record_video(gt_win, dt_win, npy_file, dt_folder, event_folder)

        flattened_gt = flattened_gt + gt_win
        flattened_dt = flattened_dt + dt_win

    return _coco_eval(flattened_gt, flattened_dt, height, width, labelmap=classes)

def evaluate_detection_RED(gt_boxes_list, dt_boxes_list, npy_file_list, dt_folder, event_folder,
                           classes=("car", "pedestrian"), height=240, width=304, time_tol=50000):
    """
    Compute detection KPIs on list of boxes in the numpy format, using the COCO python API
    https://github.com/cocodataset/cocoapi
    KPIs are only computed on timestamps where there is actual at least one box
    (fully empty frames are not considered)

    :param gt_boxes_list: list of numpy array for GT boxes (one per file)
    :param dt_boxes_list: list of numpy array for detected boxes
    :param classes: iterable of classes names
    :param height: int for box size statistics
    :param width: int for box size statistics
    :param time_tol: int size of the temporal window in micro seconds to look for a detection around a gt box
    """
    flattened_gt = []
    flattened_dt = []
    evaluator = CocoEvaluator(classes=classes, height=height, width=width,
                              verbose=True)
    for gt_boxes, dt_boxes, npy_file in zip(gt_boxes_list, dt_boxes_list, npy_file_list):

        assert np.all(gt_boxes['t'][1:] >= gt_boxes['t'][:-1])
        assert np.all(dt_boxes['t'][1:] >= dt_boxes['t'][:-1])

        all_ts = np.unique(gt_boxes['t'])
        n_steps = len(all_ts)

        gt_win, dt_win = _match_times_rev(all_ts, gt_boxes, dt_boxes, time_tol)
        if event_folder != None and record:
            record_video(gt_win, dt_win, npy_file, dt_folder, event_folder)
        flattened_gt = flattened_gt + gt_win
        flattened_dt = flattened_dt + dt_win

        if len(gt_win) == 0: # TODO: Why no GT?
            continue
        evaluator.partial_eval([np.concatenate(gt_win)], [np.concatenate(dt_win)])
    coco_kpi = evaluator.accumulate()
    for k, v in coco_kpi.items():
        print(k, ': ', v)
        print(f'coco_metrics/{k}', v)
    print('test_acc', coco_kpi['mean_ap'])

    return _coco_eval(flattened_gt, flattened_dt, height, width, labelmap=classes)

def _match_times_rev(all_ts, gt_boxes, dt_boxes, time_tol):
    windowed_gt = []
    windowed_dt = []

    dt_ts = np.unique(dt_boxes['t'])

    for ts in all_ts:
        windowed_gt.append(gt_boxes[gt_boxes['t'] == ts])

        # nearest neighbor search
        dist = np.abs(dt_ts - ts)
        nn_idx = np.argmin(dist)
        nn_ts = dt_ts[nn_idx]
        if dist[nn_idx] < time_tol:
            windowed_dt.append(dt_boxes[dt_boxes['t'] == nn_ts])
            windowed_dt[-1] = windowed_dt[-1][nms(windowed_dt[-1], windowed_dt[-1]['class_confidence'], iou_thresh=0.5)]
            windowed_dt[-1]['t'] = ts
        else:
            windowed_dt.append(dt_boxes[:0])

    return windowed_gt, windowed_dt

def _match_times(all_ts, gt_boxes, dt_boxes, time_tol):
    """
    match ground truth boxes and ground truth detections at all timestamps using a specified tolerance
    return a list of boxes vectors
    """
    gt_size = len(gt_boxes)
    dt_size = len(dt_boxes)

    windowed_gt = []
    windowed_dt = []

    low_gt, high_gt = 0, 0
    low_dt, high_dt = 0, 0
    for ts in all_ts:

        while low_gt < gt_size and gt_boxes[low_gt]['t'] < ts:
            low_gt += 1
        # the high index is at least as big as the low one
        high_gt = max(low_gt, high_gt)
        while high_gt < gt_size and gt_boxes[high_gt]['t'] <= ts:
            high_gt += 1

        # detection are allowed to be inside a window around the right detection timestamp
        low = ts - time_tol
        high = ts + time_tol
        while low_dt < dt_size and dt_boxes[low_dt]['t'] < low:
            low_dt += 1
        # the high index is at least as big as the low one
        high_dt = max(low_dt, high_dt)
        while high_dt < dt_size and dt_boxes[high_dt]['t'] <= high:
            high_dt += 1

        windowed_gt.append(gt_boxes[low_gt:high_gt])
        windowed_dt.append(dt_boxes[low_dt:high_dt])

    return windowed_gt, windowed_dt


def _coco_eval(gts, detections, height, width, labelmap=("car", "pedestrian")):
    """simple helper function wrapping around COCO's Python API
    :params:  gts iterable of numpy boxes for the ground truth
    :params:  detections iterable of numpy boxes for the detections
    :params:  height int
    :params:  width int
    :params:  labelmap iterable of class labels
    """
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(labelmap)]
    dataset, results = _to_coco_format(gts, detections, categories, height=height, width=width)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def _to_coco_format(gts, detections, categories, height=240, width=304):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox in gt:
            x1, y1 = bbox['x'], bbox['y']
            w, h = bbox['w'], bbox['h']
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox['class_id']) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox in pred:

            image_result = {
                'image_id': im_id,
                'category_id': int(bbox['class_id']) + 1,
                'score': float(bbox['class_confidence']),
                'bbox': [bbox['x'], bbox['y'], bbox['w'], bbox['h']],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    return dataset, results
