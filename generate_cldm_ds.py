import subprocess
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


SPLIT_HOLD = 10000
STRAT1 = "sparsegs_triangulate"
STRAT2 = "train_only_mapper"
split_min_triangulated_points = 20
cldm_dataset_path = "./cldm_dataset"
test_iteration = 10000
save_iteration = 30000  # nếu không muốn save gauss .ply thì set cao hơn iteration là được


if os.name == "nt":
    iteration = 1
    env = "D:/conda_env/gaussian_splatting/python.exe"
    split_copy_mode = "copy"
    data_root_path = f"D:/3dgs_improved_data"
else:
    iteration = 10000
    env = "python"
    data_root_path = "./"
    split_copy_mode = "symlink"


with open("dataset_strategy.yaml", "r") as f:
    config = yaml.safe_load(f)


def create_arg(is_sparsegs_triangulate, scene_path, output_path, split_train_view, split_train_sample_mode, stride=0,
               start_pos=0):
    base_arg = [
        env, "train.py",
        "-s", scene_path,
        "-m", output_path,
        "--eval", "--disable_viewer",
        "--iteration", str(iteration),
        "--test_iterations", str(test_iteration),
        "--save_iterations", str(save_iteration),
        "--metrics_log_interval", str(0),
        "--metrics_eval_train_count", str(-1),
        "--metrics_eval_per_view", "--metrics_compute_lpips",
        "--split_train_views", str(split_train_view),
        "--split_hold", str(SPLIT_HOLD),
        "--split_train_sample_mode", split_train_sample_mode,
        "--split_copy_mode", split_copy_mode,
        "--split_colmap_matcher", "exhaustive",
        "--split_min_triangulated_points", str(split_min_triangulated_points),
        "--split_force",
        "--for_cldm",
        "--cldm_dataset_path", cldm_dataset_path
    ]

    if is_sparsegs_triangulate:
        base_arg.extend(["--split_init_policy", STRAT1])
    else:
        base_arg.extend([
            "--split_init_policy", STRAT2,
            "--cluster_stride", str(stride),
            "--cluster_start_pos", str(start_pos)
        ])

    # use colmap cpu on linux
    if os.name != "nt":
        base_arg.append("--colmap_cpu")

    return base_arg


def process_config(config: dict, dataset_name, scene_name, scene_path):
    config_args = []
    for strat in config:
        split_train_sample_mode = config[strat]["split_method"]
        if strat == "sparsegs_triangulate":
            for run in config[strat]["runs"]:
                split_train_view = run["n_views"]
                output_path = f"./output/{dataset_name}/{scene_name}_{split_train_sample_mode}_{split_train_view}views"
                run_arg = create_arg(True, scene_path, output_path,
                                     split_train_view, split_train_sample_mode)
                config_args.append(run_arg)
        elif strat == "train_only_mapper":
            for run in config[strat]["runs"]:
                split_train_view = run["n_views"]
                stride = run["stride"]
                start_pos = run["start_pos"]
                output_path = f"./output/{dataset_name}/{scene_name}_{split_train_sample_mode}_{split_train_view}views"
                run_arg = create_arg(False, scene_path, output_path,
                                     split_train_view, split_train_sample_mode, stride, start_pos)
                config_args.append(run_arg)

    return config_args


args = []


for dataset_name in config:
    ds_config = config[dataset_name]

    default_config = ds_config["default_config"]
    not_include = ds_config["not_include"]
    test = ds_config["test"]

    for scene_name in ds_config["scene"]:
        if scene_name in not_include:
            continue

        scene_config = ds_config["scene"][scene_name]
        if scene_name in test:
            scene_path = os.path.join(data_root_path, "3dgs_dataset", "_test_", dataset_name, scene_name)
        else:
            scene_path = os.path.join(data_root_path, "3dgs_dataset", "_train_", dataset_name, scene_name)

        if scene_config == "default":  # nếu là default thì chạy theo default config của cả dataset
            config_args = process_config(default_config, dataset_name, scene_name, scene_path)
            args.extend(config_args)
        else:
            config_args = process_config(scene_config, dataset_name, scene_name, scene_path)
            args.extend(config_args)


total = len(args)
for i, sub_run in enumerate(args, 1):
    print("================================================================================")
    print(f"[{i}/{total}] Running: {' '.join(sub_run)}")
    subprocess.run(sub_run)
    print("================================================================================")
