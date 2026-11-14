# Non-Intrusive Load Monitoring (NILM) using ANNs

ANN-based system that uses whole-house smart meter readings to predict individual
appliance ON/OFF states and power consumption.

## Dataset
Pecan Street Dataport — Austin, Texas 1-minute interval data.

## Models
- **MLP**: Baseline feedforward neural network
- **LSTM**: Advanced recurrent neural network

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py
```

This runs the full pipeline: data loading, cleaning, labeling, splitting,
windowing, training (MLP + LSTM for 3 appliances), evaluation, and plot generation.

## Project Structure

```
├── src/
│   ├── config.py           # All constants, paths, hyperparameters
│   ├── data_loader.py       # Load and filter raw CSV
│   ├── data_cleaning.py     # Clean timestamps, handle missing values
│   ├── labeling.py          # ON/OFF threshold computation and labeling
│   ├── splitting.py         # Chronological train/val/test split
│   ├── windowing.py         # Create sliding input windows
│   ├── dataset.py           # PyTorch Dataset with scaling
│   ├── training.py          # Training loop with early stopping
│   ├── evaluate.py          # Metrics computation and saving
│   ├── visualization.py     # Plot generation
│   └── models/
│       ├── mlp.py           # Baseline MLP architecture
│       └── lstm_model.py    # Advanced LSTM architecture
├── run_pipeline.py          # End-to-end pipeline runner
├── report.md                # Project report
├── requirements.txt         # Python dependencies
└── results/                 # Generated metrics, plots (after running)
```

## Results
After running the pipeline, check `results/` for:
- Per-appliance benchmark metrics (JSON)
- Confusion matrix heatmaps
- Actual vs predicted power plots
- Training loss curves
- MLP vs LSTM comparison chart

## Seed
Random seed: **42** (set in `src/config.py`)
