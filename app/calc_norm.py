"""Compute training-consistent norm stats from the full all_stocks.csv.

Reproduces CustomReturnDataset._load_and_process EXACTLY:
  - read all_stocks.csv (date,ticker,close)
  - pivot -> close [date x ticker], sort by date
  - pct_change -> arithmetic returns
  - train_end = int(T * 0.8)
  - per-series mean/std over [0, train_end), std+1e-8, series with <2 valid
    train points -> (0, 1)
  - column order in training = sorted pivot columns (alphabetical), which
    pivot_table guarantees.

Writes norm_stats.json: {"tickers": [...], "mean": [...], "std": [...]} —
this file + a fresh all_stocks.csv is ALL the inference job needs. Upload to
s3://<AWS_ACCOUNT_ID>-tactis-build/checkpoints/norm_stats.json alongside the
checkpoints (same 1302-series universe as base + RL 09-10 checkpoints).

Run on the machine holding the FULL dataset (workstation WSL):
    conda activate tactis-trade
    python app/calc_norm.py --ckpt outputs/<run>/custom_tactis_medium.pth

If the checkpoint was trained on a different universe, pass the matching csv.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

TRAIN_FRACTION = 0.8  # must match cfg/dataset/custom.yaml splits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/custom-data/raw/all_stocks.csv")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint to match universe size against")
    ap.add_argument("--out", default="norm_stats.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_num = ck["model"]["flow_series_encoder.weight"].shape[0]
    print("checkpoint num_series =", ckpt_num)

    df = pd.read_csv(args.csv, usecols=["date", "ticker", "close"])
    close = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    ret = close.pct_change(fill_method=None).iloc[1:]
    train_end = int(len(ret) * TRAIN_FRACTION)
    print(f"data: {close.shape[1]} series, {len(ret)} return rows, "
          f"train_end={train_end}")

    tickers = list(ret.columns)
    means, stds = [], []
    for t in tickers:
        v = ret[t].iloc[:train_end].dropna()
        if len(v) < 2:
            means.append(0.0)
            stds.append(1.0)
            continue
        means.append(float(v.mean()))
        stds.append(float(v.std()) + 1e-8)

    with open(args.out, "w") as f:
        json.dump({"tickers": tickers, "mean": means, "std": stds}, f)

    if len(tickers) != ckpt_num:
        print(f"WARNING: data tickers {len(tickers)} != ckpt {ckpt_num} "
              f"— inference will fail its universe check")
    else:
        print(f"universe size matches checkpoint ✔")
    print(f"wrote {args.out} ({len(tickers)} series)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
