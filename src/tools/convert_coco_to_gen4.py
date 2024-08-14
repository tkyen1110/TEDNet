import os
import numpy as np
import pycocotools.coco as coco
from numpy.lib import recfunctions as rfn

ANN_DIR = '../../data/gen4/annotations'
SPLITS = ['train', 'test', 'val']
BBOX_DTYPE = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), 
              ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4'), 
              ('visibility', '<f4'), ('vx', '<f4'), ('vy', '<f4')]
DELTA_T = 50000

def reformat_anns(anns):
  video_name = None
  output = np.zeros((len(anns),), dtype=BBOX_DTYPE)
  
  for i, ann in enumerate(anns):
    if i > 0:
      assert video_name == ann['video_name']
    video_name = ann['video_name']
    
    frame_id = ann['frame_ind']
    bbox = ann['bbox']
    pre_bbox = ann['pre_bbox']
    category_id = ann['category_id']
    class_confidence = ann['class_confidence']
    track_id = ann['track_id']
    visibility_ratio = ann['visibility_ratio']

    if len(pre_bbox) > 0:
      pre_x, pre_y, pre_w, pre_h = pre_bbox
      prev_ct = [pre_x + pre_w/2, pre_y + pre_h/2]
      x, y, w, h = bbox
      ct = [x + w/2, y + h/2]
      v = [prev_ct[0]-ct[0], prev_ct[1]-ct[1]]
    else:
      v = [0, 0]

    output[i]['t'] = frame_id * DELTA_T
    output[i]['x'] = bbox[0]
    output[i]['y'] = bbox[1]
    output[i]['w'] = bbox[2]
    output[i]['h'] = bbox[3]
    output[i]['class_id'] = category_id
    output[i]['class_confidence'] = class_confidence
    output[i]['track_id'] = track_id
    output[i]['visibility'] = visibility_ratio
    output[i]['vx'] = v[0]
    output[i]['vy'] = v[1]

  label_name = os.path.splitext(video_name)[0] + "_bbox.npy"
  return label_name, output

if __name__ == '__main__':
  for split in SPLITS:
    coco_ann_path = os.path.join(ANN_DIR, 'tracking_{}.json'.format(split))
    gen4_ann_dir = os.path.join(ANN_DIR, 'tracking_{}'.format(split))
    os.makedirs(gen4_ann_dir, exist_ok=True)

    self_coco = coco.COCO(coco_ann_path)
    # self_coco.dataset.keys() = dict_keys(['images', 'annotations', 'categories', 'videos'])
    image_ids = self_coco.getImgIds()
    prev_label_name = None
    reformat_results = []
    for image_id in image_ids:
      ann_ids = self_coco.getAnnIds(imgIds=[image_id])
      anns = self_coco.loadAnns(ids=ann_ids)
      # anns[0].keys() = dict_keys(['image_id', 'category_id', 'bbox', 'pre_bbox', 'track_id', 
      #                             'occlusion', 'visibility_ratio', 'pre_visibility_ratio', 
      #                             'frame_ind', 'video_name', 'id', 'pre_id'])
      if len(anns) == 0:
        continue
      label_name, rfanns = reformat_anns(anns)
      if label_name==prev_label_name or prev_label_name==None:
        prev_label_name = label_name
      else:
        reformat_results = rfn.stack_arrays(reformat_results, usemask=False)
        gt_npy_path = os.path.join(gen4_ann_dir, prev_label_name)
        np.save(gt_npy_path, reformat_results)
        reformat_results = []
        prev_label_name = label_name

      reformat_results.append(rfanns)

    reformat_results = rfn.stack_arrays(reformat_results, usemask=False)
    gt_npy_path = os.path.join(gen4_ann_dir, prev_label_name)
    np.save(gt_npy_path, reformat_results)
