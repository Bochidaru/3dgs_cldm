#!/bin/bash

python train.py
  -s "{path_đến_data_full}"
  -m "{path_đầu_ra_output}"
  --eval
  --disable_viewer
  --iterations 10000
  --test_iterations 10000
  --save_iterations 10000
  --metrics_log_interval 0
  --metrics_eval_train_count -1
  --metrics_eval_per_view
  --metrics_compute_lpips
  --split_train_views 12
  --split_hold 8
  --split_train_sample_mode paper_even
  --split_copy_mode symlink
  --split_init_policy sparsegs_triangulate
  --split_colmap_matcher exhaustive
  --split_min_triangulated_points 100
  --split_force
  --colmap_cpu # In Linux, default colmap was not built with gpu
  --cldm_dataset_path ./cldm_dataset/