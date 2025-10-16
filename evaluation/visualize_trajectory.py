"""Basic trajectory visualizer.

Usage:
    python evaluation/visualize_trajectory.py --input path/to/trajectory.csv --output logs/trajectory.png

If no input is provided, the script will look for `logs/ppo_medarot/monitor.csv` or `logs/ppo_medarot/trajectory.csv`.
The input CSV is expected to have columns `x` and `y` (or `pos_x`, `pos_y`).

Note: Stable Baselines' Monitor CSVs do not include position by default; you must
emit a custom CSV with player coordinates or preprocess logs to include x,y columns.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_LOG = Path(__file__).resolve().parents[1] / "logs" / "ppo_medarot" / "monitor.csv"


def find_coords(df: pd.DataFrame):
    for xcol in ("x", "pos_x", "player_x"):
        for ycol in ("y", "pos_y", "player_y"):
            if xcol in df.columns and ycol in df.columns:
                return df[xcol].to_numpy(), df[ycol].to_numpy()
    # fallback: look for first two numeric columns (useful for preprocessed logs)
    numeric = df.select_dtypes("number")
    if numeric.shape[1] >= 2:
        return numeric.iloc[:, 0].to_numpy(), numeric.iloc[:, 1].to_numpy()
    raise ValueError("No coordinate columns found in dataframe")


def plot_trajectory(x, y, out_path: Path):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.scatter(x[0], y[0], color="green", label="start")
    plt.scatter(x[-1], y[-1], color="red", label="end")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend()
    plt.title("Agent trajectory")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", help="Input CSV with coordinates (x,y)")
    p.add_argument("--output", "-o", default="logs/trajectory.png", help="Output PNG path")
    args = p.parse_args()

    in_path = Path(args.input) if args.input else DEFAULT_LOG
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return

    df = pd.read_csv(in_path)
    try:
        x, y = find_coords(df)
    except Exception as e:
        print("Failed to extract coordinates:", e)
        return

    out_path = Path(args.output)
    plot_trajectory(x, y, out_path)
    print(f"Saved trajectory plot to {out_path}")


if __name__ == "__main__":
    main()
