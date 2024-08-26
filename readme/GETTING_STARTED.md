# Getting Started

This document provides tutorials to train and evaluate TEDNet. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

## Training

We have packed all the training scripts in the [experiments](../experiments) folder. Each model is trained on 2 Nvidia 3090 GPUs with 24GB of memory where the pretrained weight relies on the existing [model](https://tri-ml-public.s3.amazonaws.com/github/permatrack/pd_17fr_21ep_vis.pth) trained on PD dataset and provided by PermaTrack. We provide our models in the ablation study [here](https://drive.google.com/drive/folders/11UIkJl7z0MKRFlNyGkS6sMS39aJbFSdc?usp=sharing).

The following are the scripts to train all of the models in the ablation study.
| Methods                                      | Scripts                                                    |
| ----                                         | ----                                                       |
| CenterTrack                                  | ./centertrack_train.sh                                     |
| CenterTrack + D3D                            | ./centertrack_train.sh --dcn_3d_aggregation                |
| CenterTrack + C3D                            | ./centertrack_train.sh --conv_3d_aggregation               |
| PermaTrack                                   | ./permatrack_train.sh                                      |
| PermaTrack  + D3D                            | ./permatrack_train.sh --dcn_3d_aggregation                 |
| PermaTrack  + C3D                            | ./permatrack_train.sh --conv_3d_aggregation                |
| PermaTrack  + L<sub>con</sub>                | ./permatrack_train.sh --consistency                        |
| PermaTrack  + C3D + L<sub>con</sub>          | ./permatrack_train.sh --conv_3d_aggregation --consistency  |
| PermaTrack  + D3D + L<sub>con</sub> (TEDNet) | ./permatrack_train.sh --dcn_3d_aggregation --consistency   |

## Testing

The following are the scripts to test all of the models in the ablation study.
| Methods                                      | Scripts                                                    |
| ----                                         | ----                                                       |
| CenterTrack                                  | ./centertrack_test.sh [model_id]                                     |
| CenterTrack + D3D                            | ./centertrack_test.sh [model_id] --dcn_3d_aggregation                |
| CenterTrack + C3D                            | ./centertrack_test.sh [model_id] --conv_3d_aggregation               |
| PermaTrack                                   | ./permatrack_test.sh [model_id]                                      |
| PermaTrack  + D3D                            | ./permatrack_test.sh [model_id] --dcn_3d_aggregation                 |
| PermaTrack  + C3D                            | ./permatrack_test.sh [model_id] --conv_3d_aggregation                |
| PermaTrack  + L<sub>con</sub>                | ./permatrack_test.sh [model_id] --consistency                        |
| PermaTrack  + C3D + L<sub>con</sub>          | ./permatrack_test.sh [model_id] --conv_3d_aggregation --consistency  |
| PermaTrack  + D3D + L<sub>con</sub> (TEDNet) | ./permatrack_test.sh [model_id] --dcn_3d_aggregation --consistency   |

## Evaluation
