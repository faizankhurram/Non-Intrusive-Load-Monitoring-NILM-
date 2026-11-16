"""
Central configuration for the NILM project.
All constants, paths, and hyperparameters are defined here.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Data selection (see problems_findings.txt — Improvement Round #4)
# ---------------------------------------------------------------------------
REGION = "Austin"
# 661, 1642, 4373 have an EV charger; the rest have clean fridge/furnace signals.
# All 8 have car1 (zero where absent), furnace1 and refrigerator1.
SELECTED_HOMES = [661, 1642, 2335, 2361, 2818, 3456, 4031, 4373]

# EV charger replaces air1 (no AC usage in the chronological test split)
TARGET_APPLIANCES = ["car1", "furnace1", "refrigerator1"]

# grid is net of rooftop solar; whole-house consumption = grid + solar + solar2
SOLAR_COLUMNS = ["solar", "solar2"]
KEEP_COLUMNS = ["dataid", "localminute", "grid"] + SOLAR_COLUMNS + TARGET_APPLIANCES

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MAX_GAP_FILL = 5          # forward-fill gaps up to this many minutes
WINDOW_SIZE = 60           # number of 1-minute timesteps per input window
WINDOW_STEP = 5            # stride between windows

# ---------------------------------------------------------------------------
# ON/OFF thresholds (watts) – domain-knowledge values, checked on train only
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "car1": 1000.0,         # idle <50W, charging ~3.3kW
    "furnace1": 300.0,      # idle <50W, fan-only ~200W, heating 400W+
    "refrigerator1": 50.0,  # compressor on at 100-170W, standby <50W
}

# ---------------------------------------------------------------------------
# Train / Val / Test split ratios (chronological)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4        # AdamW (decoupled)
EPOCHS = 30
WARMUP_EPOCHS = 2
PATIENCE = 6               # early stopping on validation Macro-F1
MIN_DELTA = 1e-4
CLASS_WEIGHT_FACTOR = 1.0  # reduced — focal loss handles imbalance
MAX_POS_WEIGHT = 5.0       # cap pos_weight to prevent model collapse
REG_LOSS_WEIGHT = 0.1      # power regression on std-normalized targets

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
MODEL_TAGS = ["mlp", "lstm", "gru", "transformer"]

MLP_HIDDEN = [256, 128, 64]
DROPOUT = 0.2              # MLP + recurrent models

RNN_HIDDEN = 128           # shared by LSTM and GRU
RNN_LAYERS = 2

TF_D_MODEL = 64
TF_HEADS = 4
TF_LAYERS = 3
TF_FF = 256
TF_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
USE_STAT_FEATURES = True   # append mean, std, max, min, slope to window
N_STAT_FEATURES = 5        # number of extra statistical features

# Identifies the prepared data; cached windows/results from a different
# configuration are rebuilt instead of being reused.
DATA_SIGNATURE = "|".join([
    REGION,
    ",".join(map(str, SELECTED_HOMES)),
    ",".join(TARGET_APPLIANCES),
    ",".join(f"{a}={DEFAULT_THRESHOLDS[a]}" for a in TARGET_APPLIANCES),
    f"w{WINDOW_SIZE}s{WINDOW_STEP}",
    "aggregate=grid+solar",
    "target=last",
])
