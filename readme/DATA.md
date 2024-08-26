# Dataset Preparation

### 1 Megapixel Automotive Detection Dataset (Gen4 Dataset)
This is the first large-scale and high-resolution (1280 $\times$ 720) event-based object detection dataset for automotive scenarios provided by [Prophesee](https://www.prophesee.ai).

- Download the raw event streams (dat file) together with raw annotations (npy file) under this [link](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/).

- After downloading, copy the contents into `$TEDNet_ROOT/data/gen4`.

- The resulting data structure of raw data look like:
  ~~~
  ${TEDNet_ROOT}
  |-- data
  `-- |-- gen4
      `-- |-- train_dat
          |   |-- raw_event_streams_name.dat
          |   |-- raw_annotations_name.npy
          |   |-- ...
          |-- val_dat
          |   |-- raw_event_streams_name.dat
          |   |-- raw_annotations_name.npy
          |   |-- ...
          |-- test_dat
          |   |-- raw_event_streams_name.dat
          |   |-- raw_annotations_name.npy
          |   |-- ...
  ~~~

- Preprocess the raw event streams (dat file) to the *Event Volume* representation (h5 file) by [generate_hdf5.py](https://docs.prophesee.ai/3.1.2/metavision_sdk/modules/ml/data_processing/precomputing_features_hdf5_datasets.html) from [Prophesee Metavision SDK](https://docs.prophesee.ai/3.1.2/index.html).
  ~~~
  python3 generate_hdf5.py \
    ${TEDNet_ROOT}/data/gen4/train_dat/*.dat \
    --preprocess event_cube_paper --store_as_uint8 \
    -o ${TEDNet_ROOT}/data/gen4/train \
    --height_width 360 640 --num-workers 8
  ~~~

- The resulting data structure of event volume look like:
  ~~~
  ${TEDNet_ROOT}
  |-- data
  `-- |-- gen4
      `-- |-- train
          |   |-- event_volume_name.h5
          |   |-- ...
          |-- val
          |   |-- event_volume_name.h5
          |   |-- ...
          |-- test
          |   |-- event_volume_name.h5
          |   |-- ...
  ~~~

### Noisy Ground Truth (Noisy GT) of Testing Set
- The noisy GT comes from the training code ([train_detection.py](https://docs.prophesee.ai/3.1.2/metavision_sdk/modules/ml/quick_start/index.html#training)) of [RED model](https://papers.nips.cc/paper/2020/file/c213877427b46fa96cff6c39e837ccee-Paper.pdf) from [Prophesee Metavision SDK](https://docs.prophesee.ai/3.1.2/index.html). In a nutshell, the labeling frequency in the raw annotations is 60 Hz, i.e. 16.67 ms per frame, but the sampling period for both RED and our TEDNet is 50 ms. Therefore, there are many highly overlapping bounding boxes for the same object, which needs to be filtered out.

- The procedure for how to use [Prophesee Metavision SDK](https://docs.prophesee.ai/3.1.2/index.html) to get the noisy GT from [train_detection.py](https://docs.prophesee.ai/3.1.2/metavision_sdk/modules/ml/quick_start/index.html#training) is a little bit complicated, but it guarantees that our TEDNet is comparable with the [RED model](https://papers.nips.cc/paper/2020/file/c213877427b46fa96cff6c39e837ccee-Paper.pdf).

- The noisy GT is provied [here](https://drive.google.com/drive/folders/11Q8kAbiu-4MafIuL6Oe_tmGQi-Nto5tc?usp=sharing).

### Clean Ground Truth (Clean GT)
- Clean the raw annotations by the proposed auto-labeling algorithm
  ~~~
  cd ${TEDNet_ROOT}/src/tools
  python3 convert_gen4_to_coco.py
  ~~~

- The resulting data structure of clean GT in COCO format look like:
  ~~~
  ${TEDNet_ROOT}
  |-- data
  `-- |-- gen4
      `-- |-- annotations
          |   |-- tracking_train.json
          |   |-- tracking_val.json
          |   |-- tracking_test.json
  ~~~

- Convert the clean GT from COCO format to gen4 format for evaluation purposes.
  ~~~
  cd ${TEDNet_ROOT}/src/tools
  python3 convert_coco_to_gen4.py
  ~~~

- The resulting data structure of clean GT in gen4 format look like:
  ~~~
  ${TEDNet_ROOT}
  |-- data
  `-- |-- gen4
      `-- |-- annotations
          |   |-- tracking_train
          |   |   |-- clean_annotations_name.npy
          |   |   |-- ...
          |   |-- tracking_val
          |   |   |-- clean_annotations_name.npy
          |   |   |-- ...
          |   |-- tracking_test
          |   |   |-- clean_annotations_name.npy
          |   |   |-- ...
  ~~~

## References
Please cite the corresponding References if you use the datasets.

~~~
  @article{perot2020learning,
    title={Learning to detect objects with a 1 megapixel event camera},
    author={Perot, Etienne and De Tournemire, Pierre and Nitti, Davide and Masci, Jonathan and Sironi, Amos},
    journal={Advances in Neural Information Processing Systems},
    volume={33},
    pages={16639--16652},
    year={2020}
  }
~~~
