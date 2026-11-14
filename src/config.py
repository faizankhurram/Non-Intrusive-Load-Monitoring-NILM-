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

# Homes with the richest sub-metered coverage (from exploration)
SELECTED_HOMES = [1642, 2818, 2361, 3456, 4031]

# Three target appliances
TARGET_APPLIANCES = ["air1", "refrigerator1", "dishwasher1"]

# Columns to keep when loading the raw CSV
KEEP_COLUMNS = ["dataid", "localminute", "grid"] + TARGET_APPLIANCES

# Home reserved entirely for unseen-home testing (last in list)
UNSEEN_HOME = 4031

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MAX_GAP_FILL = 5          # forward-fill gaps up to this many minutes
WINDOW_SIZE = 60           # number of 1-minute timesteps per input window
WINDOW_STEP = 5            # stride between windows (1 = full overlap, 5 = every 5th)

# ---------------------------------------------------------------------------
# ON/OFF thresholds (watts) – set from training-data analysis
# These are initial values; labeling.py refines them from the train split.
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "air1": 100.0,
    "refrigerator1": 50.0,
    "dishwasher1": 10.0,
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
PATIENCE = 4              # early-stopping patience
CLASS_WEIGHT_FACTOR = 2.0  # weight for minority (ON) class in BCE loss

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
MLP_HIDDEN = [128, 64]
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
DROPOUT = 0.3
