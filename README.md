# Non-Intrusive Load Monitoring (NILM) using ANNs

ANN-based system that uses whole-house smart meter readings to predict individual
appliance ON/OFF states and power consumption.

## Dataset
Pecan Street Dataport — Austin, Texas 1-minute interval data (2018).

- Homes: 661, 1642, 2335, 2361, 2818, 3456, 4031, 4373
- Appliances: EV charger (`car1`), furnace (`furnace1`), refrigerator (`refrigerator1`)
- Input: whole-house consumption = `grid + solar + solar2` (grid is net of rooftop solar)
- Windows: previous 60 minutes, label = appliance state at the last minute of the window

The raw CSV must be at `1minute_data_austin/1minute_data_austin.csv`.

## Models
- **MLP**: Baseline feedforward neural network
- **LSTM**: 2-layer recurrent network
- **GRU**: 2-layer recurrent network (same structure as the LSTM)
- **Transformer**: Compact BERT4NILM-style encoder (conv embedding + 3 self-attention layers)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py
```

This runs the full pipeline: data loading, cleaning, labeling, splitting,
windowing, training (4 models for 3 appliances), evaluation, and plot generation.
Press Ctrl+C during training to pause; re-run to resume from the last checkpoint.

## Project Structure

```
├── src/
│   ├── config.py            # All constants, paths, hyperparameters
│   ├── data_loader.py       # Load and filter raw CSV
│   ├── data_cleaning.py     # Continuous timeline, gap filling, solar-corrected aggregate
│   ├── labeling.py          # ON/OFF threshold computation and labeling
│   ├── splitting.py         # Chronological train/val/test split
│   ├── windowing.py         # Create sliding input windows
│   ├── dataset.py           # PyTorch Dataset with scaling
│   ├── training.py          # Training loop with early stopping on val Macro-F1
│   ├── evaluate.py          # Metrics computation and saving
│   ├── visualization.py     # Plot generation
│   └── models/
│       ├── weight_init.py        # Weight / output-bias initialization helpers
│       ├── mlp.py                # Baseline MLP
│       ├── recurrent_base.py     # Shared LSTM/GRU architecture
│       ├── lstm_model.py         # LSTM
│       ├── gru_model.py          # GRU
│       └── transformer_model.py  # Transformer encoder
├── run_pipeline.py          # End-to-end pipeline runner
├── problems_findings.txt    # Log of problems, experiments and changes
├── report.md                # Project report
├── requirements.txt         # Python dependencies
└── results/                 # Generated metrics, plots (after running)
```

## Results
After running the pipeline, check `results/` for:
- Per-appliance benchmark metrics (JSON)
- Confusion matrix heatmaps
- Actual vs predicted power plots
- Training loss / validation Macro-F1 curves
- Model comparison chart with CCP benchmark lines

## Seed
Random seed: **42** (set in `src/config.py`)
