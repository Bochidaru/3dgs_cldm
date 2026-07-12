import os
import importlib
import sys
import random
import hashlib
import yaml
import subprocess
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from cldm_dataset_sampling_strategy import GROUP_MAPPING, RULES, TEST_SPLIT
sys.stdout.reconfigure(line_buffering=True)


STATUS = {0: "PENDING", 1: "RUNNING", 2: "COMPLETED", 3: "FAILED"}


root = r"D:\3dgs_improved_data\3dgs_dataset"
config_name = "runtime_config.yaml"
limit_try = 5
SPLIT_HOLD = 10000
STRAT1 = "sparsegs_triangulate"
STRAT2 = "train_only_mapper"
split_min_triangulated_points = 20
cldm_dataset_path = "./cldm_dataset"
test_iteration = 30000
save_iteration = 30000  # nếu không muốn save gauss .ply thì set cao hơn iteration là được
colmap_gpu = True

if os.name == "nt":
    iteration = 1
    env = "D:/conda_env/gaussian_splatting/python.exe"
    split_copy_mode = "copy"
    root = f"D:/3dgs_improved_data/3dgs_dataset"
else:
    iteration = 10000
    env = "python"
    root = "./3dgs_dataset"
    split_copy_mode = "symlink"


def get_rng(dataset_name, scene_name):
    key = f"{dataset_name}/{scene_name}"

    seed = int(
        hashlib.md5(key.encode()).hexdigest(),
        16
    ) % (2 ** 32)

    return random.Random(seed)


def count_images(scene_path):
    images_dir = os.path.join(scene_path, "images")

    if not os.path.isdir(images_dir):
        return 0

    return len([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ])


def find_matching_rule(dataset_name, image_count):
    group = GROUP_MAPPING[dataset_name]

    for rule in RULES[group]:

        if rule["image_count_min"] <= image_count <= rule["image_count_max"]:
            return group, rule

    raise RuntimeError(
        "No rule found for dataset_name={} image_count={}".format(dataset_name, image_count)
    )


def build_scene_config(
        dataset_name,
        scene_name,
        image_count
):
    # Is Test?
    isTest = scene_name in TEST_SPLIT.get(dataset_name, [])

    # Tìm rule tương ứng với dataset và lượng ảnh
    group, rule = find_matching_rule(
        dataset_name,
        image_count
    )

    scene_config = {
        "group": group,
        "isTest": isTest,
        "image_count": image_count,
        "cldm_gt_ratio": rule.get("cldm_gt_ratio", "full"),
        "sparsegs_triangulate": [],
        "train_only_mapper": []
    }

    rng = get_rng(dataset_name, scene_name)

    for run in rule["sparsegs_triangulate"]:
        scene_config["sparsegs_triangulate"].append({
            "split_method": run["split_method"],

            # random candidate
            "n_views": rng.choice(run["candidate_views"]),
            "status": STATUS[0]
        })

    for idx, run in enumerate(rule["train_only_mapper"]):
        if idx == 0:
            start_pos = 0
        else:
            start_pos = 70
        scene_config["train_only_mapper"].append({
            "split_method": run["split_method"],

            "n_views": rng.choice(run["candidate_views"]),

            "stride": run["candidate_strides"][0],

            "start_pos": start_pos,
            "status": STATUS[0]
        })

    return scene_config


def build_runtime_config(root, old_runtime_config: dict):
    runtime_config = {}

    datasets = sorted(os.listdir(root))

    for dataset_name in datasets:
        dataset_path = os.path.join(root, dataset_name)

        if os.path.isdir(dataset_path):
            if dataset_name not in GROUP_MAPPING:
                print(f"[SKIP] {dataset_name} not found in GROUP_MAPPING")
                continue

            if (old_runtime_config is not None and dataset_name in old_runtime_config):
                runtime_config[dataset_name] = old_runtime_config[dataset_name]
            
            else:
                runtime_config[dataset_name] = {}
                for scene_name in sorted(os.listdir(dataset_path)):
                    scene_path = os.path.join(
                        dataset_path,
                        scene_name
                    )
                    image_count = count_images(scene_path)
                    runtime_config[dataset_name][scene_name] = build_scene_config(dataset_name, scene_name, image_count)

    return runtime_config


def save_config(config, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"Saved config: {save_path}")


def load_config(config_path):
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_config(runtime_config, dataset, scene, strat, candidate_id, new_status, new_nviews=None, new_stride=None):
    runtime_config[dataset][scene][strat][candidate_id]["status"] = new_status
    if new_nviews:
        runtime_config[dataset][scene][strat][candidate_id]["n_views"] = new_nviews
    if new_stride:
        runtime_config[dataset][scene][strat][candidate_id]["stride"] = new_stride
    save_config(runtime_config, os.path.join(root, config_name))


def count_total_runs(runtime_config):
    total = 0
    for dataset_n in runtime_config:
        for scene_n in runtime_config[dataset_n]:
            scene_cfg = runtime_config[dataset_n][scene_n]
            total += len(scene_cfg["sparsegs_triangulate"])
            total += len(scene_cfg["train_only_mapper"])
    return total


def run_sparsegs_until_success(base_arg, split_train_view):
    status = None
    for _ in range(limit_try):
        try:
            arg = base_arg.copy()
            arg[arg.index("--split_train_views")+1] = str(split_train_view)
            subprocess.run(arg, check=True)
            status = STATUS[2]
            break

        except subprocess.CalledProcessError as e:
            print(f"Error with n_views = {split_train_view}, retry with +3 views")
            split_train_view += 3
            status = STATUS[3]

    return status, split_train_view

def run_trainonly_until_success(base_arg, split_train_view, stride):
    status = None
    for i in range(limit_try):
        try:
            arg = base_arg.copy()
            arg[arg.index("--cluster_stride")+1] = str(stride)
            arg[arg.index("--split_train_views")+1] = str(split_train_view)
            subprocess.run(arg, check=True)
            status = STATUS[2]
            break

        except subprocess.CalledProcessError as e:
            if i < 3:
                print(f"Error with n_views = {split_train_view}, stride = {stride}, retry with -1 stride")
                stride -= 1
            if i > 2:
                print(f"Error with n_views = {split_train_view}, stride = {stride}, retry with +3 views")
                split_train_view += 3
            status = STATUS[3]

    return status, split_train_view, stride


def run_3dgs(is_sparsegs_triangulate, scene_path, output_path, isTest, cldm_gt_ratio, group,
             split_train_sample_mode, split_train_view, stride=0, start_pos=0):
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
        "--cldm_dataset_path", cldm_dataset_path,
        "--cldm_ds_group", group,
        "--cldm_gt_ratio", cldm_gt_ratio,
    ]

    if is_sparsegs_triangulate:
        base_arg.extend(["--split_init_policy", "sparsegs_triangulate"])
    else:
        base_arg.extend([
            "--split_init_policy", "train_only_mapper",
            "--cluster_stride", str(stride),
            "--cluster_start_pos", str(start_pos)
        ])

    if isTest:
        base_arg.extend(["--is_cldm_test"])

    # use colmap cpu on linux
    if not colmap_gpu:
        base_arg.append("--colmap_cpu")

    if is_sparsegs_triangulate:
        status, new_nviews = run_sparsegs_until_success(base_arg, split_train_view)
        return status, new_nviews, None
    else:
        status, new_nviews, new_stride = run_trainonly_until_success(base_arg, split_train_view, stride)
        return status, new_nviews, new_stride


if __name__ == "__main__":
    old_config = load_config(os.path.join(root, config_name))

    runtime_config = build_runtime_config(root, old_config)

    # First Save
    save_config(runtime_config, os.path.join(root, config_name))

    total_run = count_total_runs(runtime_config)
    run_idx = 0
    # Build args
    for dataset_n in runtime_config:
        for scene_n in runtime_config[dataset_n]:
            scene_p = os.path.join(root, dataset_n, scene_n)
            group = runtime_config[dataset_n][scene_n]["group"]
            image_count = runtime_config[dataset_n][scene_n]["image_count"]
            isTest = runtime_config[dataset_n][scene_n]["isTest"]
            cldm_gt_ratio = runtime_config[dataset_n][scene_n]["cldm_gt_ratio"]

            sparsegs_triangulate = runtime_config[dataset_n][scene_n]["sparsegs_triangulate"]  # List
            train_only_mapper = runtime_config[dataset_n][scene_n]["train_only_mapper"]  # List

            for candidate_id, run in enumerate(sparsegs_triangulate):
                run_idx += 1
                split_train_sample_mode = run["split_method"]
                split_train_view = run["n_views"]
                output_p = f"./output/{dataset_n}/{scene_n}_{split_train_sample_mode}_{split_train_view}views"
                print("================================================================================")
                print(f"[{run_idx}/{total_run}] Running: {dataset_n}/{scene_n}/{split_train_sample_mode}/{split_train_view}views")
                if run["status"] in [STATUS[2], STATUS[3]]:
                    print(f"This has already been completed!")
                    continue
                update_config(runtime_config, dataset_n, scene_n, STRAT1, candidate_id, STATUS[1])  # RUNNING
                start_time = time.time()
                new_status, new_nviews, _ = run_3dgs(True, scene_p, output_p, isTest, cldm_gt_ratio,
                                                  group, split_train_sample_mode, split_train_view)
                end_time = time.time()
                update_config(runtime_config, dataset_n, scene_n, STRAT1, candidate_id, new_status, new_nviews=new_nviews)
                print(f"⏱ Ran in: {end_time-start_time:.2f} second")
                print("================================================================================")

            for candidate_id, run in enumerate(train_only_mapper):
                run_idx += 1
                split_train_sample_mode = run["split_method"]
                split_train_view = run["n_views"]
                stride = run["stride"]
                start_pos = run["start_pos"]
                output_p = f"./output/{dataset_n}/{scene_n}_{split_train_sample_mode}_{split_train_view}views"
                print("================================================================================")
                print(f"[{run_idx}/{total_run}] Running: {dataset_n}/{scene_n}/{split_train_sample_mode}/{split_train_view}views/{stride}stride")
                if run["status"] in [STATUS[2], STATUS[3]]:
                    print(f"This has already been completed!")
                    continue
                update_config(runtime_config, dataset_n, scene_n, STRAT2, candidate_id, STATUS[1])  # RUNNING
                start_time = time.time()
                new_status, new_nviews, new_stride = run_3dgs(False, scene_p, output_p, isTest, cldm_gt_ratio,
                                                  group, split_train_sample_mode, split_train_view, stride, start_pos)
                end_time = time.time()
                update_config(runtime_config, dataset_n, scene_n, STRAT2, candidate_id, new_status, new_nviews=new_nviews, new_stride=new_stride)
                print(f"⏱ Ran in: {end_time - start_time:.2f} second")
                print("================================================================================")
