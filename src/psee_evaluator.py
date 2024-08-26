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
# https://github.com/hamarh/HMNet_pth/blob/main/experiments/detection/scripts/psee_evaluator.py which is modified from
# https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/psee_evaluator.py
# The list of modifications are as follows:
# (1) (HMNet)  "min_box_side" for box filtering is modified following the previous work:
#     Perot, Etienne, et al. "Learning to detect objects with a 1 megapixel event camera." Advances in Neural Information Processing Systems 33 (2020): 16639-16652.
# (2) (HMNet)  Configs for GEN1 and GEN4 are added and passed to "evaluate_detection"
# (3) (TEDNet) "evaluate_detection_RED" uses the same evaluation code as RED model

import glob
import numpy as np
import os
import argparse
import pickle as pkl
from numpy.lib import recfunctions as rfn

from coco_eval import evaluate_detection, evaluate_detection_RED
from psee_toolbox.io.box_filtering import filter_boxes
from psee_toolbox.io.box_loading import reformat_boxes
from common import get_list, mkdir

EVAL_CONF_GEN1 = dict(
    classes = ('car', 'pedestrian'),
    width = 304,
    height = 240,
    time_tol = 25000, # +/- 25 msec (50 msec)
)

EVAL_CONF_GEN4 = dict(
    classes = ('background', 'pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light'),
    width = 1280//2,
    height = 720//2,
    time_tol = 25000, # +/- 25 msec (50 msec)
)

def get_min_stats(boxes_list):
    min_confidence = 1.0
    min_class_id = np.inf
    max_class_id = -np.inf
    for boxes in boxes_list:
        min_confidence_ = np.min(boxes['class_confidence'])
        min_class_id_ = np.min(boxes['class_id'])
        max_class_id_ = np.max(boxes['class_id'])

        if min_confidence_ < min_confidence:
            min_confidence = min_confidence_
        if min_class_id_ < min_class_id:
            min_class_id = min_class_id_
        if max_class_id_ > max_class_id:
            max_class_id = max_class_id_
    return min_confidence, min_class_id, max_class_id

def evaluate_folders(dt_folder, gt_lst, discard_small_obj, event_folder, camera, dt_confidence, gt_visibility, run_nms):
    dt_file_paths = get_list(dt_folder, ext='npy')
    gt_file_paths = get_list(gt_lst, ext='npy')

    if len(dt_file_paths) != len(gt_file_paths):
      gt_file_paths = []
      for dt_file_path in dt_file_paths:
        gt_file_paths.append(dt_file_path.replace(dt_folder, gt_lst))
    
    assert len(dt_file_paths) == len(gt_file_paths)
    print("There are {} GT bboxes and {} PRED bboxes".format(len(gt_file_paths), len(dt_file_paths)))
    npy_file_list = list()
    for dt_file_path, gt_file_path in zip(dt_file_paths, gt_file_paths):
        npy_file_list.append(os.path.basename(gt_file_path))
        assert os.path.basename(dt_file_path)==os.path.basename(gt_file_path)

    print("DT: dtype_names = ", np.load(dt_file_paths[0]).dtype.names)
    result_boxes_list = [np.load(p) for p in dt_file_paths]
    min_confidence, min_class_id, max_class_id = get_min_stats(result_boxes_list)
    print("DT: min_confidence = ", min_confidence)
    print("DT: min_class_id = ", min_class_id)
    print("DT: max_class_id = ", max_class_id)
    result_boxes_list = [reformat_boxes(p, confidence=dt_confidence, run_nms=run_nms) for p in result_boxes_list]
    print("DT: dtype_names = ", result_boxes_list[0].dtype.names)
    print()
    # result_boxes_list[0].dtype.names = ('t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence')

    print("GT: dtype_names = ", np.load(gt_file_paths[0]).dtype.names)
    gt_boxes_list = [np.load(p) for p in gt_file_paths]
    min_confidence, min_class_id, max_class_id = get_min_stats(gt_boxes_list)
    print("GT: min_confidence = ", min_confidence)
    print("GT: min_class_id = ", min_class_id)
    print("GT: max_class_id = ", max_class_id)
    gt_boxes_list = [reformat_boxes(p, visibility=gt_visibility) for p in gt_boxes_list]
    print("GT: dtype_names = ", gt_boxes_list[0].dtype.names)
    print()

    # gt_boxes_list[0].dtype.names     = ('t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence')
    # if 'invalid' in gt_boxes_list[0].dtype.names:
    #     for i, gt_boxes in enumerate(gt_boxes_list):
    #         invalids = gt_boxes['invalid']
    #         if np.sum(invalids) > 0:
    #             gt_boxes = gt_boxes[np.logical_not(invalids)]
    #         gt_boxes = rfn.drop_fields(gt_boxes, 'invalid')
    #         gt_boxes_list[i] = gt_boxes

    eval_conf = EVAL_CONF_GEN4 if camera == 'GEN4' else EVAL_CONF_GEN1
    if discard_small_obj:
        min_box_diag = 60 if camera == 'GEN4' else 30
        min_box_side = 20 if camera == 'GEN4' else 10
        filter_boxes_fn = lambda x:filter_boxes(x, int(5e5), min_box_diag, min_box_side)
        gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
        result_boxes_list = map(filter_boxes_fn, result_boxes_list)
    # evaluate_detection(gt_boxes_list, result_boxes_list, npy_file_list, dt_folder, event_folder, **eval_conf)
    evaluate_detection_RED(gt_boxes_list, result_boxes_list, npy_file_list, dt_folder, event_folder, **eval_conf)

def main():
    parser = argparse.ArgumentParser(prog='psee_evaluator.py')
    parser.add_argument('gt_lst', type=str, help='Text file contaiing list of GT .npy files')
    parser.add_argument('dt_folder', type=str, help='RESULT folder containing .npy files')
    parser.add_argument('--discard_small_obj', action='store_true', default=False)
    parser.add_argument('--event_folder', type=str, help='Event folder containing .dat files')
    parser.add_argument('--camera', type=str, default='GEN4', help='GEN1 (QVGA) or GEN4 (720p)')
    parser.add_argument('--dt_confidence', type=float, default=0.1)
    parser.add_argument('--gt_visibility', type=float, default=1.0)
    parser.add_argument('--run_nms', action='store_true')
    opt = parser.parse_args()
    evaluate_folders(opt.dt_folder, opt.gt_lst, opt.discard_small_obj, opt.event_folder, opt.camera, opt.dt_confidence, opt.gt_visibility, opt.run_nms)

if __name__ == '__main__':
    main()
