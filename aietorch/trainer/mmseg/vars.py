import os


CONFIG_BASE_DIR = os.path.join(os.path.dirname(__file__), "configs")
RUNTIME_CONFIG_PATH = os.path.join(CONFIG_BASE_DIR, "_base_", "default_runtime.py")
SCHEDULE_CONFIG_PATH = os.path.join(CONFIG_BASE_DIR, "_base_", "schedules", "schedule_20k.py")

DEFAULT_CFG = {
    "seed"    : 0,
    "gpu_ids" : range(1),
    "device"  : "cuda",
}