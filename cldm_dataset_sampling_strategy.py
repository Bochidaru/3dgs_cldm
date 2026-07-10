# cldm_dataset_sampling_strategy.py

# ============================================================
# DATASET -> DIFFICULTY GROUP
# ============================================================

GROUP_MAPPING = {

    # Partial Object
    "mvimgnet": "PO",
    "DTU": "PO",
    "nerf_llff_data": "PO",
    "ThaiDuong": "PO",

    # Full Object
    "co3d": "FO",
    "uco3d": "FO",

    # Complex Object
    "mipnerf360": "CO",
    "TanksAndTemples": "CO",

    # Large Scene
    "DL3DV": "LS",
    "ScanNet++": "LS",
}


# ============================================================
# TEST SPLIT
# ============================================================

TEST_SPLIT = {
    "mvimgnet": [
        "rifle1"
    ],

    "co3d": [
        "car3",
    ],

    "DTU": [
        "scan114",
    ],

    "nerf_llff_data": [
        "horns",
    ],

    "mipnerf360": [
        "garden",
    ],

    "TanksAndTemples": [
        "Truck",
    ],

    "ThaiDuong": [
        "chair"
    ]
}


# ============================================================
# RULES
# ============================================================
#
# image_count:
#     số ảnh trong scene/images
#
# candidate_views:
#     runtime adaptation sẽ pick 1 trong các candidate_views, nếu không chạy thì +3
#
# candidate_strides:
#     runtime adaptation sẽ thử lần lượt
#
# ============================================================

RULES = {

    # ========================================================
    # Partial Object
    # ========================================================

    "PO": [

        {
            "image_count_min": 0,
            "image_count_max": 999999,
            "cldm_gt_ratio": "1/2",

            "sparsegs_triangulate": [
                {
                    "split_method": "paper_even",
                    "candidate_views": [6, 9, 12],
                }
            ],

            "train_only_mapper": []
        },
    ],

    # ========================================================
    # Full Object
    # ========================================================

    "FO": [

        {
            "image_count_min": 0,
            "image_count_max": 999999,
            "cldm_gt_ratio": "1/3",
            "sparsegs_triangulate": [
                {
                    "split_method": "paper_even",
                    "candidate_views": [9, 12, 15, 18],
                }
            ],

            "train_only_mapper": [
                {
                    "split_method": "cluster_stride",
                    "candidate_views": [12, 15],
                    "candidate_strides": [4, 3, 2, 1],
                }
            ]
        },
    ],

    # ========================================================
    # Large Scene
    # ========================================================

    "LS": [

        {
            "image_count_min": 0,
            "image_count_max": 999999,
            "cldm_gt_ratio": "1/2",

            "sparsegs_triangulate": [
                {
                    "split_method": "paper_even",
                    "candidate_views": [15, 18, 24, 27],
                }
            ],
        },
    ],

    # ========================================================
    # Complex Object
    # ========================================================

    "CO": [ 

        {
            "image_count_min": 0,
            "image_count_max": 299,
            "cldm_gt_ratio": "1/2",

            "sparsegs_triangulate": [

                {
                    "split_method": "paper_even",
                    "candidate_views": [12, 15],
                },

                {
                    "split_method": "paper_even",
                    "candidate_views": [18, 24],
                },
            ],

            "train_only_mapper": [

                {
                    "split_method": "cluster_stride",
                    "candidate_views": [12, 15],
                    "candidate_strides": [4, 3, 2, 1],
                },

                {
                    "split_method": "cluster_stride",
                    "candidate_views": [18, 24],
                    "candidate_strides": [4, 3, 2, 1],
                },
            ]
        },

        {
            "image_count_min": 300,
            "image_count_max": 399,
            "cldm_gt_ratio": "1/3",

            "sparsegs_triangulate": [

                {
                    "split_method": "pose_fps",
                    "candidate_views": [15, 18],
                },

                {
                    "split_method": "paper_even",
                    "candidate_views": [24, 27],
                },
            ],

            "train_only_mapper": [

                {
                    "split_method": "cluster_stride",
                    "candidate_views": [18],
                    "candidate_strides": [4, 3, 2, 1],
                },

                {
                    "split_method": "cluster_stride",
                    "candidate_views": [27],
                    "candidate_strides": [4, 3, 2, 1],
                },
            ]
        },

        {
            "image_count_min": 400,
            "image_count_max": 999999,
            "cldm_gt_ratio": "1/3",

            "sparsegs_triangulate": [

                {
                    "split_method": "pose_fps",
                    "candidate_views": [18, 21],
                },

                {
                    "split_method": "paper_even",
                    "candidate_views": [30, 33, 36],
                },
            ],

            "train_only_mapper": [

                {
                    "split_method": "cluster_stride",
                    "candidate_views": [21],
                    "candidate_strides": [4, 3, 2, 1],
                },

                {
                    "split_method": "cluster_stride",
                    "candidate_views": [30],
                    "candidate_strides": [4, 3, 2, 1],
                },
            ]
        },

    ]
}