"""Project paths, data settings, and training hyperparameters."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "1minute_data_austin")
RAW_CSV = os.path.join(RAW_DATA_DIR, "1minute_data_austin.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
WINDOWS_DIR = os.path.join(PROJECT_ROOT, "data", "windows")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOT_DIRS = {
    "confusion_matrix": os.path.join(RESULTS_DIR, "confusion_matrices"),
    "curves": os.path.join(RESULTS_DIR, "training_curves"),
    "class_metrics": os.path.join(RESULTS_DIR, "class_performance"),
    "dynamics": os.path.join(RESULTS_DIR, "training_dynamics"),
    "comparison": os.path.join(RESULTS_DIR, "model_comparison"),
}

for d in [PROCESSED_DIR, WINDOWS_DIR, MODELS_DIR, METRICS_DIR, *PLOT_DIRS.values()]:
    os.makedirs(d, exist_ok=True)

REGION = "Austin"
SELECTED_HOMES = [661, 1642, 2335, 2361, 2818, 3456, 4031, 4373]
TARGET_APPLIANCES = ["car1", "furnace1", "refrigerator1"]
SOLAR_COLUMNS = ["solar", "solar2"]
KEEP_COLUMNS = ["dataid", "localminute", "grid"] + SOLAR_COLUMNS + TARGET_APPLIANCES

MAX_GAP_FILL = 5
WINDOW_SIZE = 60
WINDOW_STEP = 5

DEFAULT_THRESHOLDS = {
    "car1": 1000.0,
    "furnace1": 300.0,
    "refrigerator1": 50.0,
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
WARMUP_EPOCHS = 2
PATIENCE = 6
MIN_DELTA = 1e-4
CLASS_WEIGHT_FACTOR = 1.0
MAX_POS_WEIGHT = 5.0
REG_LOSS_WEIGHT = 0.1

MODEL_TAGS = ["mlp", "lstm", "gru", "transformer"]
MLP_HIDDEN = [256, 128, 64]
DROPOUT = 0.2
RNN_HIDDEN = 128
RNN_LAYERS = 2
TF_D_MODEL = 64
TF_HEADS = 4
TF_LAYERS = 3
TF_FF = 256
TF_DROPOUT = 0.1

USE_STAT_FEATURES = True
N_STAT_FEATURES = 5

DATA_SIGNATURE = "|".join([
    REGION,
    ",".join(map(str, SELECTED_HOMES)),
    ",".join(TARGET_APPLIANCES),
    ",".join(f"{a}={DEFAULT_THRESHOLDS[a]}" for a in TARGET_APPLIANCES),
    f"w{WINDOW_SIZE}s{WINDOW_STEP}",
    "aggregate=grid+solar",
    "target=last",
])
