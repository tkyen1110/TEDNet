# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Defines some tools to handle events.
In particular :
    -> defines events' types
    -> defines functions to read events from binary .dat files using numpy
    -> defines functions to write events to binary .dat files using numpy
"""

from __future__ import print_function
import numpy as np
from numpy.lib import recfunctions as rfn

BBOX_DTYPE = np.dtype({'names':['t','x','y','w','h','class_id','track_id','class_confidence'], 'formats':['<i8','<f4','<f4','<f4','<f4','<u4','<u4','<f4'], 'offsets':[0,8,12,16,20,24,28,32], 'itemsize':40})

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

def reformat_boxes(boxes, confidence=0.1, visibility=1.0, run_nms=False):
    """ReFormat boxes according to new rule
    This allows to be backward-compatible with imerit annotation.
        't' = 'ts'
        'class_confidence' = 'confidence'
    """
    if 'visibility' in boxes.dtype.names:
        boxes = boxes[boxes['visibility'] <= visibility]
    boxes = boxes[['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence']]
    boxes = boxes[boxes['class_confidence'] >= confidence]

    if run_nms:
        new_boxes = []
        for ut in np.unique(boxes['t']):
            box = boxes[boxes['t']==ut]
            box = box[nms(box, box['class_confidence'], iou_thresh=0.5)]
            new_boxes.append(box)
        new_boxes = rfn.stack_arrays(new_boxes, usemask=False)
        return new_boxes

    return boxes
