"""
Exploratory data analysis used to choose the region, homes and appliances.

Run from the project root:
    python -m src.eda
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT, RESULTS_DIR, SELECTED_HOMES, TARGET_APPLIANCES

EDA_DIR = os.path.join(RESULTS_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

REGIONS = {
    "austin": os.path.join(PROJECT_ROOT, "1minute_data_austin", "1minute_data_austin.csv"),
    "california": os.path.join(PROJECT_ROOT, "1minute_data_california", "1minute_data_california.csv"),
    "newyork": os.path.join(PROJECT_ROOT, "1minute_data_newyork", "1minute_data_newyork.csv"),
}
PR_DIR = os.path.join(PROJECT_ROOT, "pr_1min")

CANDIDATE_APPLIANCES = [
    "air1", "car1", "furnace1", "refrigerator1", "dishwasher1",
]
PROVISIONAL_THRESHOLDS_KW = {
    "air1": 0.3,
    "car1": 1.0,
    "furnace1": 0.3,
    "refrigerator1": 0.05,
    "dishwasher1": 0.2,
}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
CHUNK = 400_000


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_region_columns(path, columns):
    frames = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK, low_memory=False):
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def chronological_on_ratios(home_df, appliance, thr_kw):
    home_df = home_df.sort_values("localminute")
    values = home_df[appliance].astype(float)
    n = len(values)
    if n < 1000:
        return None
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    parts = {
        "train": values.iloc[:train_end],
        "val": values.iloc[train_end:val_end],
        "test": values.iloc[val_end:],
    }
    ratios = {}
    for name, series in parts.items():
        s = series.dropna()
        if len(s) == 0:
            return None
        ratios[name] = float((s >= thr_kw).mean())
    return ratios


def profile_region(name, path):
    cols = ["dataid", "localminute", "grid"] + CANDIDATE_APPLIANCES
    available = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in cols if c in available]
    print(f"\nLoading {name} ({os.path.basename(path)}) ...")
    df = load_region_columns(path, cols)
    df["localminute"] = pd.to_datetime(df["localminute"], utc=True)

    homes = sorted(df["dataid"].unique().tolist())
    print(f"  homes={len(homes)}  rows={len(df):,}  "
          f"range={df['localminute'].min().date()} -> {df['localminute'].max().date()}")

    coverage = {}
    for app in CANDIDATE_APPLIANCES:
        if app not in df.columns:
            coverage[app] = 0
            continue
        present = df.groupby("dataid")[app].apply(lambda s: s.notna().mean() > 0.01)
        coverage[app] = int(present.sum())
    print("  homes with usable readings (>1% non-null):")
    for app, n in coverage.items():
        print(f"    {app:16s} {n}")

    qualify = {app: [] for app in CANDIDATE_APPLIANCES if app in df.columns}
    for home_id, home_df in df.groupby("dataid"):
        for app in list(qualify.keys()):
            thr = PROVISIONAL_THRESHOLDS_KW[app]
            ratios = chronological_on_ratios(home_df, app, thr)
            if ratios is None:
                continue
            ok = all(0.05 <= ratios[s] <= 0.95 for s in ("train", "val", "test"))
            if ok:
                qualify[app].append((int(home_id), ratios))

    print("  homes with ON ratio in [5%, 95%] in train, val AND test:")
    for app, rows in qualify.items():
        print(f"    {app:16s} {len(rows)}")
        if rows and app in ("air1", "car1", "furnace1", "refrigerator1"):
            for home_id, ratios in sorted(rows, key=lambda x: x[0])[:5]:
                print(f"      home {home_id}: "
                      f"train={ratios['train']:.3f} val={ratios['val']:.3f} "
                      f"test={ratios['test']:.3f}")
            if len(rows) > 5:
                print(f"      ... ({len(rows) - 5} more)")

    return df, qualify, coverage


def inspect_pr():
    header("1. Puerto Rico folder (pr_1min)")
    files = sorted(glob.glob(os.path.join(PR_DIR, "*.csv*")))
    print(f"We found {len(files)} measurement files (csv/csv.gz).")
    for f in files[:6]:
        print(f"  {os.path.basename(f)}")
    if len(files) > 6:
        print(f"  ... ({len(files) - 6} more)")
    print("These files are split by quantity (real power, current, THD, ...) "
          "and only cover Sep-Nov 2023.")
    print("Decision: we drop pr_1min — too short and not the same home/"
          "appliance schema as the mainland Pecan Street dumps.")


def plot_region_qualify(region_stats):
    apps = CANDIDATE_APPLIANCES
    regions = list(region_stats.keys())
    x = np.arange(len(apps))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, region in enumerate(regions):
        vals = [region_stats[region].get(app, 0) for app in apps]
        ax.bar(x + (i - 1) * width, vals, width, label=region)
    ax.set_xticks(x)
    ax.set_xticklabels(apps, rotation=15)
    ax.set_ylabel("Qualifying homes (train/val/test)")
    ax.set_title("Homes with stable ON/OFF balance across chronological splits")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "region_qualifying_homes.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")


def austin_deep_dive(df, qualify):
    header("4. Austin deep dive — why air1 fails and what we keep")

    print("\nair1 ON ratios on a few homes (shows the seasonal test collapse):")
    sample_homes = [661, 1642, 2361, 2818, 3456, 4373, 4767, 8156]
    for home_id in sample_homes:
        home_df = df[df["dataid"] == home_id]
        if home_df.empty or "air1" not in home_df.columns:
            continue
        ratios = chronological_on_ratios(home_df, "air1", PROVISIONAL_THRESHOLDS_KW["air1"])
        if ratios is None:
            continue
        print(f"  home {home_id}: train={ratios['train']:.3f}  "
              f"val={ratios['val']:.3f}  test={ratios['test']:.3f}")

    print("\ncar1 / furnace1 / refrigerator1 qualifying home counts in Austin:")
    for app in ("car1", "furnace1", "refrigerator1"):
        print(f"  {app}: {len(qualify.get(app, []))}")

    print("\nNegative grid readings (rooftop solar export):")
    solar_homes = []
    for home_id, home_df in df.groupby("dataid"):
        grid = home_df["grid"].astype(float)
        neg = float((grid < 0).mean())
        if neg > 0.05:
            solar_homes.append((int(home_id), neg))
    solar_homes.sort(key=lambda x: -x[1])
    print(f"  homes with >5% negative grid: {len(solar_homes)}")
    for home_id, neg in solar_homes[:8]:
        print(f"    home {home_id}: {100 * neg:.1f}% of minutes grid < 0")
    print("  Decision: treat whole-house consumption as grid + solar (+ solar2).")


def plot_power_histograms(df):
    header("5. Power distributions and thresholds (Austin)")
    apps = ["car1", "furnace1", "refrigerator1", "air1"]
    thr = PROVISIONAL_THRESHOLDS_KW
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, app in zip(axes.ravel(), apps):
        vals = df[app].astype(float).dropna()
        vals = vals[(vals >= 0) & (vals < vals.quantile(0.995))]
        ax.hist(vals, bins=60, color="steelblue", alpha=0.85)
        ax.axvline(thr[app], color="red", linestyle="--",
                   label=f"threshold={thr[app]} kW")
        ax.set_title(app)
        ax.set_xlabel("Power (kW)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle("Austin appliance power histograms (provisional ON thresholds)")
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "austin_power_histograms.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")
    print("We set car1=1.0 kW, furnace1=0.3 kW, refrigerator1=0.05 kW "
          "(later stored in watts in config).")


def choose_homes(df, qualify):
    header("6. Home selection")

    car_homes = {h for h, _ in qualify.get("car1", [])}
    furn_homes = {h for h, _ in qualify.get("furnace1", [])}
    fridge_homes = {h for h, _ in qualify.get("refrigerator1", [])}
    present = set(int(x) for x in df["dataid"].unique())

    print(f"car1 qualifying homes: {sorted(car_homes)}")
    print(f"furnace1 qualifying:   {len(furn_homes)}")
    print(f"refrigerator1 qualifying: {len(fridge_homes)}")

    print("We keep three EV homes (661, 1642, 4373) for car1 and fill with "
          "homes that have clean furnace and fridge signals:")
    print(f"  {SELECTED_HOMES}")
    missing = [h for h in SELECTED_HOMES if h not in present]
    if missing:
        raise SystemExit(f"Selected homes missing from Austin CSV: {missing}")

    print("\nPer-home ON ratios on the final appliances:")
    for home_id in SELECTED_HOMES:
        home_df = df[df["dataid"] == home_id]
        bits = []
        for app in TARGET_APPLIANCES:
            thr = PROVISIONAL_THRESHOLDS_KW[app]
            ratios = chronological_on_ratios(home_df, app, thr)
            if ratios is None:
                bits.append(f"{app}=n/a")
            else:
                bits.append(f"{app} testON={ratios['test']:.3f}")
        print(f"  {home_id}: " + ", ".join(bits))

    print("\nFinal appliances:", TARGET_APPLIANCES)
    print("We replaced air1 with car1 because air1 has almost no ON samples "
          "in the chronological test split (Nov-Dec).")


def plot_final_balance(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(len(SELECTED_HOMES))
    width = 0.25
    splits = ("train", "val", "test")
    colors = ("tab:blue", "tab:orange", "tab:green")

    for ax, app in zip(axes, TARGET_APPLIANCES):
        for i, (split, color) in enumerate(zip(splits, colors)):
            vals = []
            for home_id in SELECTED_HOMES:
                home_df = df[df["dataid"] == home_id]
                ratios = chronological_on_ratios(
                    home_df, app, PROVISIONAL_THRESHOLDS_KW[app]
                )
                vals.append(ratios[split] if ratios else 0.0)
            ax.bar(x + (i - 1) * width, vals, width, label=split, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(SELECTED_HOMES, rotation=45)
        ax.set_title(app)
        ax.set_ylim(0, 1)
        ax.axhline(0.05, color="red", linestyle=":", linewidth=1)
        ax.axhline(0.95, color="red", linestyle=":", linewidth=1)
    axes[0].set_ylabel("ON ratio")
    axes[0].legend(fontsize=8)
    fig.suptitle("ON ratios for selected homes (train / val / test)")
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "selected_homes_on_ratios.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    header("EDA — dataset and appliance selection")
    print("We explore the available Pecan Street folders, then narrow the "
          "choice to the region, homes and appliances used in this project.")

    inspect_pr()

    header("2. Mainland regions — coverage under a chronological split")
    region_qualify_counts = {}
    austin_df = None
    austin_qualify = None

    # Profile the other regions first so we do not hold them with Austin in memory.
    order = ["california", "newyork", "austin"]
    for name in order:
        path = REGIONS[name]
        if not os.path.exists(path):
            print(f"Missing {path} — skip")
            continue
        df, qualify, _ = profile_region(name, path)
        region_qualify_counts[name] = {app: len(rows) for app, rows in qualify.items()}
        if name == "austin":
            austin_df = df
            austin_qualify = qualify
        else:
            del df

    plot_region_qualify(region_qualify_counts)

    header("3. Region decision")
    print("Austin has the most homes where furnace1 and refrigerator1 stay "
          "balanced in train, val and test, and it is the only region with "
          "several usable EV chargers (car1).")
    print("California and New York have many refrigerators but almost no "
          "usable furnace1 / car1 under the same split.")
    print("Decision: we use Austin.")

    if austin_df is None:
        raise SystemExit("Austin data was not loaded.")

    austin_deep_dive(austin_df, austin_qualify)
    plot_power_histograms(austin_df)
    choose_homes(austin_df, austin_qualify)
    plot_final_balance(austin_df)

    header("EDA complete")
    print(f"Plots written to {EDA_DIR}")
    print("Selected region=Austin")
    print(f"Selected homes={SELECTED_HOMES}")
    print(f"Selected appliances={TARGET_APPLIANCES}")


if __name__ == "__main__":
    main()
