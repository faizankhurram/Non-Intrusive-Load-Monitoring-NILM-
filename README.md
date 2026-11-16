# Non-Intrusive Load Monitoring (NILM)

We predict individual appliance **ON/OFF** state and **power draw** from whole-house
smart meter readings only, using four neural models trained on Pecan Street
1-minute data.

---

## Dataset

| | |
|---|---|
| **Source** | Pecan Street Dataport — Austin, Texas (2018) |
| **Homes** | `661`, `1642`, `2335`, `2361`, `2818`, `3456`, `4031`, `4373` |
| **Appliances** | EV charger (`car1`), furnace (`furnace1`), refrigerator (`refrigerator1`) |
| **Input** | Whole-house consumption = `grid + solar + solar2` |
| **Windows** | 60 minutes, stride 5 · label = state at the **last** minute of the window |
| **Split** | Chronological 70% / 15% / 15% per home |

Place the raw CSV at:

```text
1minute_data_austin/1minute_data_austin.csv
```

We chose this region, home set and appliance trio after exploring all available
folders (`python -m src.eda`). Plots from that analysis go to `results/eda/`.

---

## Models

| Model | Role |
|---|---|
| **MLP** | Baseline feedforward network on the flattened window |
| **LSTM** | 2-layer recurrent model |
| **GRU** | Same structure as the LSTM, with a GRU cell |
| **Transformer** | Compact encoder (conv embedding + 3 self-attention layers) |

Shared training setup: AdamW, focal loss + light power regression, selection and
early stopping on validation Macro-F1, 30 epochs, seed **42**.

**Min Benchmark:** Accuracy ≥ 0.85 · Balanced Accuracy ≥ 0.75 · Macro-F1 ≥ 0.70

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run

**Full pipeline** (load → clean → label → split → window → train → evaluate → plots):

```bash
python run_pipeline.py
```

Press `Ctrl+C` once during training to pause; run the same command again to resume
from the last checkpoint.

| Task | Command |
|---|---|
| Dataset / appliance EDA | `python -m src.eda` |
| Regenerate plots only | `python -m src.visualization` |

---

## Project structure

```text
├── src/
│   ├── config.py              # Paths, homes, appliances, hyperparameters
│   ├── eda.py                 # Exploratory analysis for data selection
│   ├── data_loader.py         # Load and filter the raw CSV
│   ├── data_cleaning.py       # Timeline gaps, solar-corrected aggregate
│   ├── labeling.py            # ON/OFF thresholds and labels
│   ├── splitting.py           # Chronological train / val / test split
│   ├── windowing.py           # Sliding windows
│   ├── dataset.py             # PyTorch Dataset + scaling
│   ├── training.py            # Train loop, early stopping, checkpoints
│   ├── evaluate.py            # Test metrics
│   ├── visualization.py       # Result plots
│   └── models/
│       ├── weight_init.py
│       ├── mlp.py
│       ├── recurrent_base.py  # Shared LSTM / GRU body
│       ├── lstm_model.py
│       ├── gru_model.py
│       └── transformer_model.py
├── run_pipeline.py
├── problems_findings.txt      # Problems, experiments, decisions
├── requirements.txt
├── models/                    # Weights (*.pt) and training histories
└── results/                   # Metrics and plots (see below)
```

---

## Results layout

After a pipeline run, `results/` is organised by output type
(`{model}_{appliance}_...`):

| Folder | Contents |
|---|---|
| `eda/` | Plots from dataset / appliance selection |
| `metrics/` | Test metrics JSON (Acc, BalAcc, Macro-F1, MAE, RMSE, confusion matrix) |
| `confusion_matrices/` | Raw confusion matrix heatmaps |
| `class_performance/` | Row-normalised CM + per-class precision / recall / F1 |
| `training_curves/` | Train / val loss with validation Macro-F1 |
| `training_dynamics/` | Generalisation gap, LR schedule, selected epoch |
| `model_comparison/` | All models vs the Min Benchmark lines |

Weights and `*_history.json` files live under `models/`.
