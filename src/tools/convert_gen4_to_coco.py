from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import os
import copy
import cv2
import h5py
import math
from queue import Queue
from metavision_core.event_io import EventNpyReader

ANN_DIR = '../../data/gen4/annotations'
os.makedirs(ANN_DIR, exist_ok=True)
SPLITS = ['train', 'test', 'val']
BBOX_DTYPE = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), 
              ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')]
BBOX_DTYPE_NAME = ('t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence', 'track_id')

def get_bbox_element(bbox):
  assert bbox.dtype == BBOX_DTYPE or bbox.dtype.names == BBOX_DTYPE_NAME
  t = bbox['t']
  x = bbox['x']
  y = bbox['y']
  w = bbox['w']
  h = bbox['h']
  class_id = bbox['class_id']
  class_confidence = bbox['class_confidence']
  track_id = bbox['track_id']
  assert class_id>=0
  assert class_confidence >= 0 and class_confidence < 1
  assert track_id>=0
  return t, x, y, w, h, class_id, class_confidence, track_id

def filter_bboxes(bboxes, shape, downsample):
  assert bboxes.dtype == BBOX_DTYPE or bboxes.dtype.names == BBOX_DTYPE_NAME
  c, h, w = shape
  filtered_bboxes = np.empty(0, dtype=bboxes.dtype)
  track_ids = sorted(set(bboxes['track_id']))
  for track_id in track_ids:
    t = 0
    x1 = np.inf
    y1 = np.inf
    x2 = 0
    y2 = 0
    class_confidence = 0
    votes_table = {}
    for bbox in bboxes[bboxes['track_id']==track_id]:
      _t, _x, _y, _w, _h, _class_id, _class_confidence, _track_id = get_bbox_element(bbox)
      t = np.maximum(t, _t)
      x1 = np.minimum(x1, _x)
      y1 = np.minimum(y1, _y)
      x2 = np.maximum(x2, _x + _w)
      y2 = np.maximum(y2, _y + _h)
      class_confidence = np.maximum(class_confidence, _class_confidence)
      if _class_id in votes_table:    # Check if key in hash table
          votes_table[_class_id] += 1 # Increment counter
      else:
          votes_table[_class_id] = 1  # Create counter for vote

    x1 = np.clip(x1/downsample, 0, w-1)
    y1 = np.clip(y1/downsample, 0, h-1)
    x2 = np.clip(x2/downsample, 0, w-1)
    y2 = np.clip(y2/downsample, 0, h-1)
    class_id = max(votes_table, key=votes_table.get)
    if x2-x1 > 0.0 and y2-y1 > 0.0:
      _boxes = np.array([(t, x1, y1, x2-x1, y2-y1, class_id, class_confidence, track_id)], dtype=bboxes.dtype)
      filtered_bboxes = np.concatenate((filtered_bboxes, _boxes), axis=0)
  assert filtered_bboxes.dtype == BBOX_DTYPE or filtered_bboxes.dtype.names == BBOX_DTYPE_NAME
  return filtered_bboxes

def get_overlapped_bboxes(gt_boxes, i):
    ti, xi, yi, wi, hi, class_id_i, confidence_i, track_id_i = get_bbox_element(gt_boxes[i])
    overlapped_bboxes = []

    for j, gt_box in enumerate(gt_boxes):
        tj, xj, yj, wj, hj, class_id_j, confidence_j, track_id_j = get_bbox_element(gt_boxes[j])
        if track_id_j == track_id_i:
            continue
        x1 = np.max([xi, xj])
        y1 = np.max([yi, yj])
        x2 = np.min([xi+wi, xj+wj])
        y2 = np.min([yi+hi, yj+hj])
        if x1 < x2 and y1 < y2:
            overlapped_bboxes.append([int(x1)-int(xi), int(y1)-int(yi), int(x2)-int(xi), int(y2)-int(yi)])

    return overlapped_bboxes  
  
# "categories"
cats = ['Pedestrian', 'Two_Wheeler', 'Car', 'Truck', 'Bus', 'Traffic_Sign',  'Traffic_Light']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

queue_size = 2

if __name__ == '__main__':
  for split in SPLITS:
    ret = {'images': [], 'annotations': [], "categories": cat_info,
           'videos': []}
    num_images = 0

    event_h5_dir = "../../data/gen4/{}".format(split)     # .h5
    gt_label_dir = "../../data/gen4/{}_dat".format(split) # _bbox.npy

    for i, video_name in enumerate(os.listdir(event_h5_dir)):
      label_name = os.path.splitext(video_name)[0] + "_bbox.npy"
      ret['videos'].append({'id': i, 'file_name': video_name})

      video_path = os.path.join(event_h5_dir, video_name)
      label_path = os.path.join(gt_label_dir, label_name)

      gt_label = EventNpyReader(label_path)
      gt_label.seek_time(0)

      with h5py.File(video_path, 'r') as rfile:
        t, c, h5_height, h5_width = rfile['data'].shape
        h5_data = rfile['data'][0]
        h5_data_attrs = rfile['data'].attrs
        delta_t = h5_data_attrs['delta_t']
        event_input_height = h5_data_attrs['event_input_height']
        event_input_width = h5_data_attrs['event_input_width']
        shape = h5_data_attrs['shape']
        store_as_uint8 = h5_data_attrs['store_as_uint8']
        downsample = event_input_height//h5_height

        image_files = list(range(t))
        num_images_video = len(image_files)
        image_range = [0, num_images_video - 1]
        print(i+1, ': ', 'num_frames', image_range[1] - image_range[0] + 1, video_name)
        pre_gt_boxes = None
        pre_gt_visibility_ratios = []
        pre_gt_ann_ids = []
        still_hits = {}
        # visibility status of every track
        track_vis = {}
        # minimal number of consequitive visible frames after which to start supervising behind occlusions
        visibility_range = 2
        for j, image_name in enumerate(image_files):
          if (j < image_range[0] or j > image_range[1]):
            continue
          num_images += 1
          image = rfile['data'][j]
          occupancy_mask = (np.sum(image != 0, axis=0) > 0)
          image_info = {'file_name': '{}/{}'.format(split, video_name),
                        'id': num_images-1,
                        'video_id': i,
                        'frame_id': j}
          
          ret['images'].append(image_info)

          gt_boxes = gt_label.load_delta_t(delta_t=delta_t)
          gt_boxes = filter_bboxes(gt_boxes, shape=shape, downsample=downsample)
          gt_ann_ids = []
          gt_track_ids = []
          gt_visibility_ratios = []
          for k, gt_box in enumerate(gt_boxes):
            _t, _x, _y, _w, _h, _class_id, _class_confidence, _track_id = get_bbox_element(gt_box)
            _cx = _x + _w / 2
            _cy = _y + _h / 2
            _xi, _yi, _wi, _hi = int(_x), int(_y), int(_w), int(_h)
            gt_track_ids.append(_track_id)
            mask = np.ones((_hi, _wi), dtype=bool)
            overlapped_bboxes = get_overlapped_bboxes(gt_boxes, k)
            for overlapped_bbox in overlapped_bboxes:
                x1, y1, x2, y2 = overlapped_bbox
                mask[y1:y2, x1:x2] = False
            occ_true = np.sum((occupancy_mask[_yi:_yi+_hi, _xi:_xi+_wi]==True)*mask)
            occ_false = np.sum((occupancy_mask[_yi:_yi+_hi, _xi:_xi+_wi]==False)*mask)
            if occ_true + occ_false > 0: 
                occ_rate = occ_true / (occ_true + occ_false)
            else:
                occ_rate = 0

            still_object = False
            pre_bbox = []
            if isinstance(pre_gt_boxes, np.ndarray):
              if np.any(pre_gt_boxes['track_id']==_track_id):
                pre_gt_box = pre_gt_boxes[pre_gt_boxes['track_id']==_track_id]
                assert len(pre_gt_box) == 1
                pre_t, pre_x, pre_y, pre_w, pre_h, pre_class_id, pre_confidence, pre_track_id = get_bbox_element(pre_gt_box[0])
                pre_bbox = [pre_x, pre_y, pre_w, pre_h]
                pre_cx = pre_x + pre_w / 2
                pre_cy = pre_y + pre_h / 2
                # print(pre_cx , _cx, _w, pre_cy, _cy, _h)
                velocity = np.array([(pre_cx-_cx)/_w, (pre_cy-_cy)/_h])
                vel = math.sqrt(np.sum(velocity**2))

                if _track_id not in still_hits.keys():
                  still_hits[_track_id] = 0

                if vel < 0.03 and occ_rate < 0.1:
                  still_object = True
                  if still_hits[_track_id] < 5:
                      still_hits[_track_id] = still_hits[_track_id] + 1
                elif still_hits[_track_id] > 0:
                  still_object = True
                  still_hits[_track_id] = still_hits[_track_id] - 1
                else:
                  still_object = False
            elif occ_rate < 0.1:
              if _track_id not in still_hits.keys():
                still_hits[_track_id] = 0
              still_object = True
              if still_hits[_track_id] < 5:
                  still_hits[_track_id] = still_hits[_track_id] + 1

            # object is visible unless it is still
            visible = not still_object
            if visible:
              gt_visibility_ratios.append(1.0)
            else:
              gt_visibility_ratios.append(0.0)
            if isinstance(pre_gt_boxes, np.ndarray) and _track_id in pre_gt_boxes['track_id']:
              idx = pre_gt_boxes['track_id'].tolist().index(_track_id)
              pre_gt_visibility_ratio = pre_gt_visibility_ratios[idx]
              pre_gt_ann_id = pre_gt_ann_ids[idx]
            else:
              pre_gt_visibility_ratio = -1.0
              pre_gt_ann_id = -1
            ann = { 'image_id': num_images-1,
                    'category_id': int(_class_id)+1,
                    'class_confidence': float(_class_confidence),
                    'bbox': [float(_x), float(_y), float(_w), float(_h)],
                    'pre_bbox': [float(bbox) for bbox in pre_bbox],
                    'track_id': int(_track_id),
                    'occlusion': 0.0 if still_object else 1.0,
                    'visibility_ratio': 0.0 if still_object else 1.0,
                    'pre_visibility_ratio': pre_gt_visibility_ratio,
                    'frame_ind': j,
                    'video_name': video_name,
                    'id': int(len(ret['annotations'])),
                    'pre_id': pre_gt_ann_id }
            gt_ann_ids.append(ann['id'])
            if _track_id not in track_vis.keys():
              track_vis[_track_id] = Queue(maxsize=queue_size)
            if visible:
              ret['annotations'].append(ann)
              if track_vis[_track_id].qsize() == queue_size:
                track_vis[_track_id].get(block=False)
              track_vis[_track_id].put(j)
            else:
              # if track_vis[_track_id].qsize() == queue_size and (track_vis[_track_id].queue[0] != j-2 or track_vis[_track_id].queue[1] != j-1):
              #   print("track_id = {}; time = {} ms; {} / {} ms".format(_track_id, j*50, track_vis[_track_id].queue[0]*50, track_vis[_track_id].queue[1]*50))
              if track_vis[_track_id].qsize() == queue_size and track_vis[_track_id].queue[0] == j-2 and track_vis[_track_id].queue[1] == j-1:
                ret['annotations'].append(ann)
                track_vis[_track_id].get(block=False)
                track_vis[_track_id].put(j)

          if isinstance(pre_gt_boxes, np.ndarray):
            for pre_gt_box in pre_gt_boxes:
              pre_t, pre_x, pre_y, pre_w, pre_h, pre_class_id, pre_confidence, pre_track_id = get_bbox_element(pre_gt_box)
              if pre_track_id not in gt_track_ids and pre_track_id in still_hits.keys():
                del still_hits[pre_track_id]

          pre_gt_boxes = gt_boxes
          pre_gt_visibility_ratios = gt_visibility_ratios
          pre_gt_ann_ids = gt_ann_ids

    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    out_path = '{}/tracking_{}.json'.format(ANN_DIR, split)
    json.dump(ret, open(out_path, 'w'))
