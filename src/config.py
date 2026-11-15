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

for d in [PROCESSED_DIR, WINDOWS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------
REGION = "Austin"
SELECTED_HOMES = [1642, 2818, 2361, 3456, 4767, 4031]

# Three target appliances (furnace1 replaces dishwasher1 — see findings)
TARGET_APPLIANCES = ["air1", "refrigerator1", "furnace1"]

KEEP_COLUMNS = ["dataid", "localminute", "grid"] + TARGET_APPLIANCES

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MAX_GAP_FILL = 5          # forward-fill gaps up to this many minutes
WINDOW_SIZE = 60           # number of 1-minute timesteps per input window
WINDOW_STEP = 5            # stride between windows

# ---------------------------------------------------------------------------
# ON/OFF thresholds (watts) – domain-knowledge values
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "air1": 300.0,          # clear gap between <50W (off) and 500W+ (on)
    "refrigerator1": 50.0,  # compressor on at 50-200W, standby <50W
    "furnace1": 200.0,      # fan idle <50W, heating active 200W+
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
EPOCHS = 20
PATIENCE = 4
CLASS_WEIGHT_FACTOR = 1.0  # reduced — focal loss handles imbalance
MAX_POS_WEIGHT = 5.0       # cap pos_weight to prevent model collapse

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
MLP_HIDDEN = [256, 128, 64]   # deeper MLP
LSTM_HIDDEN = 128              # more capacity
LSTM_LAYERS = 1
DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
USE_STAT_FEATURES = True   # append mean, std, max, min, slope to window
N_STAT_FEATURES = 5        # number of extra statistical features
