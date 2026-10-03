import os
import glob
import pandas as pd
import statistics
from pathlib import Path

base = Path(__file__).resolve().parent / "sign_lang_detection_augmented"

for batch in [8, 16, 32]:
    rows = []

    for seed in [42, 17, 119]:
        matches = sorted(base.glob(f"*seed_{seed}_batch_{batch}*"))
        valid_dirs = [d for d in matches if d.is_dir() and (d / "results.csv").exists()]

        if not valid_dirs:
            continue

        # pick the newest valid run
        folder = max(valid_dirs, key=lambda d: d.stat().st_mtime)
        csv_path = folder / "results.csv"
        df = pd.read_csv(csv_path)

        if "metrics/mAP50-95(B)" not in df.columns:
            continue

        best_idx = df["metrics/mAP50-95(B)"].idxmax()

        row = {
            "seed": seed,
            "best_epoch": int(df.loc[best_idx, "epoch"]),
            "runtime_s": float(df.loc[best_idx, "time"]),
        }
        rows.append(row)

    if not rows:
        print(f"Batch {batch}: no valid runs found")
        continue

    runtimes = [r["runtime_s"] for r in rows]
    mean_t = statistics.mean(runtimes)
    std_t = statistics.stdev(runtimes)

    print(f"Batch {batch}:")
    for r in rows:
        print(f"  seed={r['seed']}, best_epoch={r['best_epoch']}, runtime={r['runtime_s']:.1f}s")
    print(f"  mean runtime = {mean_t:.2f}s")
    print(f"  sample SD = {std_t:.2f}s")