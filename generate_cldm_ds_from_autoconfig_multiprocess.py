import os
import sys
import random
import hashlib
import yaml
import subprocess
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from cldm_dataset_sampling_strategy import GROUP_MAPPING, RULES, TEST_SPLIT
import multiprocessing as mp
from filelock import FileLock
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
save_iteration = 30000
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

# ---- NEW: multiprocessing + NVIDIA MPS config (1 GPU dùng chung) ----
GPU_ID = 0
NUM_WORKERS = 3  # số job train.py chạy ĐỒNG THỜI trên cùng 1 GPU qua MPS, tuỳ VRAM mà chỉnh

# % SM dành riêng cho mỗi worker (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE).
# None = để MPS tự chia sẻ công bằng giữa các client (mặc định, khuyên dùng trước).
# Đặt số (vd 100 // NUM_WORKERS) nếu muốn ép chia đều, tránh 1 job "ăn" hết SM.
MPS_ACTIVE_THREAD_PERCENTAGE = None

MPS_PIPE_DIR = "/tmp/nvidia-mps"
MPS_LOG_DIR = "/tmp/nvidia-log"

CONFIG_PATH = os.path.join(root, config_name)
CONFIG_LOCK_PATH = CONFIG_PATH + ".lock"


def get_rng(dataset_name, scene_name):
    key = f"{dataset_name}/{scene_name}"
    seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)
    return random.Random(seed)


def count_images(scene_path):
    images_dir = os.path.join(scene_path, "images")
    if not os.path.isdir(images_dir):
        return 0
    return len([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])


def find_matching_rule(dataset_name, image_count):
    group = GROUP_MAPPING[dataset_name]
    for rule in RULES[group]:
        if rule["image_count_min"] <= image_count <= rule["image_count_max"]:
            return group, rule
    raise RuntimeError(
        "No rule found for dataset_name={} image_count={}".format(dataset_name, image_count)
    )


def build_scene_config(dataset_name, scene_name, image_count):
    isTest = scene_name in TEST_SPLIT.get(dataset_name, [])
    group, rule = find_matching_rule(dataset_name, image_count)

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
            "n_views": rng.choice(run["candidate_views"]),
            "status": STATUS[0]
        })

    for idx, run in enumerate(rule["train_only_mapper"]):
        start_pos = 0 if idx == 0 else 70
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

            if old_runtime_config is not None and dataset_name in old_runtime_config:
                runtime_config[dataset_name] = old_runtime_config[dataset_name]
            else:
                runtime_config[dataset_name] = {}
                for scene_name in sorted(os.listdir(dataset_path)):
                    scene_path = os.path.join(dataset_path, scene_name)
                    image_count = count_images(scene_path)
                    runtime_config[dataset_name][scene_name] = build_scene_config(
                        dataset_name, scene_name, image_count
                    )

    return runtime_config


def save_config(config, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Saved config: {save_path}")


def load_config(config_path):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_config_locked(dataset, scene, strat, candidate_id, new_status,
                          new_nviews=None, new_stride=None):
    """
    Đọc lại config MỚI NHẤT từ đĩa, sửa đúng 1 entry, ghi lại.
    Bắt buộc phải reload trước khi ghi vì nhiều process cùng ghi 1 file -
    nếu giữ bản copy cũ trong RAM sẽ ghi đè mất update của process khác.
    """
    with FileLock(CONFIG_LOCK_PATH):
        cfg = load_config(CONFIG_PATH)
        entry = cfg[dataset][scene][strat][candidate_id]
        entry["status"] = new_status
        if new_nviews:
            entry["n_views"] = new_nviews
        if new_stride:
            entry["stride"] = new_stride
        save_config(cfg, CONFIG_PATH)
        return cfg


def read_status_locked(dataset, scene, strat, candidate_id):
    with FileLock(CONFIG_LOCK_PATH):
        cfg = load_config(CONFIG_PATH)
        return cfg[dataset][scene][strat][candidate_id]["status"]


def count_total_runs(runtime_config):
    total = 0
    for dataset_n in runtime_config:
        for scene_n in runtime_config[dataset_n]:
            scene_cfg = runtime_config[dataset_n][scene_n]
            total += len(scene_cfg["sparsegs_triangulate"])
            total += len(scene_cfg["train_only_mapper"])
    return total


def build_job_list(runtime_config):
    """Làm phẳng toàn bộ (dataset, scene, strategy, candidate) thành 1 list job độc lập."""
    jobs = []
    for dataset_n, scenes in runtime_config.items():
        for scene_n, scene_cfg in scenes.items():
            common = dict(
                dataset=dataset_n,
                scene=scene_n,
                group=scene_cfg["group"],
                isTest=scene_cfg["isTest"],
                cldm_gt_ratio=scene_cfg["cldm_gt_ratio"],
            )

            for candidate_id, run in enumerate(scene_cfg["sparsegs_triangulate"]):
                jobs.append({
                    **common,
                    "strat": STRAT1,
                    "candidate_id": candidate_id,
                    "is_sparsegs": True,
                    "split_method": run["split_method"],
                    "n_views": run["n_views"],
                })

            for candidate_id, run in enumerate(scene_cfg["train_only_mapper"]):
                jobs.append({
                    **common,
                    "strat": STRAT2,
                    "candidate_id": candidate_id,
                    "is_sparsegs": False,
                    "split_method": run["split_method"],
                    "n_views": run["n_views"],
                    "stride": run["stride"],
                    "start_pos": run["start_pos"],
                })
    return jobs


def run_sparsegs_until_success(base_arg, split_train_view):
    status = None
    for _ in range(limit_try):
        try:
            arg = base_arg.copy()
            arg[arg.index("--split_train_views") + 1] = str(split_train_view)
            subprocess.run(arg, check=True)
            status = STATUS[2]
            break
        except subprocess.CalledProcessError:
            print(f"Error with n_views = {split_train_view}, retry with +3 views")
            split_train_view += 3
            status = STATUS[3]
    return status, split_train_view


def run_trainonly_until_success(base_arg, split_train_view, stride):
    status = None
    for i in range(limit_try):
        try:
            arg = base_arg.copy()
            arg[arg.index("--cluster_stride") + 1] = str(stride)
            arg[arg.index("--split_train_views") + 1] = str(split_train_view)
            subprocess.run(arg, check=True)
            status = STATUS[2]
            break
        except subprocess.CalledProcessError:
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

    if not colmap_gpu:
        base_arg.append("--colmap_cpu")

    if is_sparsegs_triangulate:
        status, new_nviews = run_sparsegs_until_success(base_arg, split_train_view)
        return status, new_nviews, None
    else:
        status, new_nviews, new_stride = run_trainonly_until_success(base_arg, split_train_view, stride)
        return status, new_nviews, new_stride


# ---- NEW: NVIDIA MPS daemon management (chạy trong main process) ----
def _mps_control(command):
    return subprocess.run(
        ["nvidia-cuda-mps-control"],
        input=command + "\n",
        text=True,
        capture_output=True,
        env=os.environ,
    )


def start_mps_daemon():
    os.makedirs(MPS_PIPE_DIR, exist_ok=True)
    os.makedirs(MPS_LOG_DIR, exist_ok=True)
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE_DIR
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = MPS_LOG_DIR
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    result = _mps_control("get_server_list")
    if result.returncode == 0 and result.stdout.strip():
        print("[MPS] control daemon đã chạy sẵn, dùng lại.")
        return

    print("[MPS] khởi động nvidia-cuda-mps-control daemon...")
    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True, env=os.environ)
        time.sleep(1)
    except FileNotFoundError:
        raise RuntimeError(
            "Không tìm thấy 'nvidia-cuda-mps-control'. Cần cài driver NVIDIA hỗ trợ MPS "
            "và (khuyên dùng) chạy trước: sudo nvidia-smi -c EXCLUSIVE_PROCESS"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Không khởi động được MPS daemon: {e}")


def stop_mps_daemon():
    print("[MPS] tắt daemon...")
    _mps_control("quit")


# ---- NEW: worker-side MPS setup ----
def _worker_init(percent_queue):
    """
    Chạy đúng 1 lần khi Pool khởi tạo mỗi worker process.
    Tất cả worker đều trỏ vào CÙNG 1 GPU (0) - MPS daemon lo việc time-slice/ghép
    nhiều context CUDA của các subprocess train.py chạy song song trên GPU đó.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE_DIR
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = MPS_LOG_DIR

    percent = percent_queue.get()
    if percent is not None:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percent)

    print(f"[WORKER pid={os.getpid()}] MPS GPU {GPU_ID}, active_thread_pct={percent}")


def run_job(job):
    """Chạy 1 job độc lập (dataset/scene/strategy/candidate). Được gọi trong worker process."""
    dataset_n = job["dataset"]
    scene_n = job["scene"]
    strat = job["strat"]
    candidate_id = job["candidate_id"]

    # kiểm tra lại trạng thái mới nhất phòng khi đã có worker khác hoàn thành job này trước đó
    current_status = read_status_locked(dataset_n, scene_n, strat, candidate_id)
    if current_status in (STATUS[2], STATUS[3]):
        print(f"[SKIP] {dataset_n}/{scene_n}/{strat}/{candidate_id} already {current_status}")
        return job, current_status

    scene_p = os.path.join(root, dataset_n, scene_n)
    split_train_view = job["n_views"]
    split_train_sample_mode = job["split_method"]
    output_p = f"./output/{dataset_n}/{scene_n}_{split_train_sample_mode}_{split_train_view}views"

    print("================================================================================")
    print(f"[pid={os.getpid()}] Running: {dataset_n}/{scene_n}/{strat}/{split_train_sample_mode}/{split_train_view}views")

    update_config_locked(dataset_n, scene_n, strat, candidate_id, STATUS[1])  # RUNNING

    start_time = time.time()
    if job["is_sparsegs"]:
        new_status, new_nviews, new_stride = run_3dgs(
            True, scene_p, output_p, job["isTest"], job["cldm_gt_ratio"], job["group"],
            split_train_sample_mode, split_train_view
        )
    else:
        new_status, new_nviews, new_stride = run_3dgs(
            False, scene_p, output_p, job["isTest"], job["cldm_gt_ratio"], job["group"],
            split_train_sample_mode, split_train_view, job["stride"], job["start_pos"]
        )
    end_time = time.time()

    update_config_locked(dataset_n, scene_n, strat, candidate_id, new_status,
                          new_nviews=new_nviews, new_stride=new_stride)

    print(f"⏱ [pid={os.getpid()}] Ran in: {end_time - start_time:.2f} second")
    print("================================================================================")
    return job, new_status


if __name__ == "__main__":
    old_config = load_config(CONFIG_PATH)
    runtime_config = build_runtime_config(root, old_config)

    # First Save
    save_config(runtime_config, CONFIG_PATH)

    all_jobs = build_job_list(runtime_config)
    pending_jobs = [
        j for j in all_jobs
        if runtime_config[j["dataset"]][j["scene"]][j["strat"]][j["candidate_id"]]["status"]
        not in (STATUS[2], STATUS[3])
    ]

    total_run = count_total_runs(runtime_config)
    print(f"Total jobs: {total_run}, pending: {len(pending_jobs)}")

    if not pending_jobs:
        print("Nothing to do.")
        sys.exit(0)

    num_workers = NUM_WORKERS
    manager = mp.Manager()
    percent_queue = manager.Queue()
    for _ in range(num_workers):
        percent_queue.put(MPS_ACTIVE_THREAD_PERCENTAGE)

    start_mps_daemon()
    try:
        # processes=num_workers -> Pool tạo đúng num_workers process cố định,
        # tất cả cùng chạy trên GPU 0 nhưng được MPS ghép context để chạy song song.
        with mp.Pool(processes=num_workers, initializer=_worker_init, initargs=(percent_queue,)) as pool:
            for job, status in pool.imap_unordered(run_job, pending_jobs):
                print(f"[DONE] {job['dataset']}/{job['scene']}/{job['strat']}/{job['candidate_id']} -> {status}")
    finally:
        stop_mps_daemon()

    print("ALL JOBS FINISHED")