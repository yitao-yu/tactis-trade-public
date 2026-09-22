"""Inference: checkpoint + fresh data -> target weights JSON.

Daily Batch-job path: download checkpoint from S3 (entrypoint), load fresh
price data, sample forward returns from the TACTiS model, convert samples to
target weights via the same allocator pipeline used in validation, write a
weights JSON consumable by run_live.py / pipeline.run_venue.

Dry-run by design: this script only produces weights; execution stays in
run_live.py behind LiveGuard.

Usage (inside the Batch container):
    python3 /opt/app/infer.py \
        --checkpoint /opt/model/checkpoint.pth \
        --data /opt/data/all_stocks.csv \
        --out /opt/app/weights.json

Standalone smoke (CPU, tiny subset):
    python3 app/infer.py --checkpoint <pth> --data <csv> --out /tmp/w.json \
        --series-limit 8 --num-samples 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HIST_LEN = 21
PRED_LEN = 4
NUM_SAMPLES = 64          # MC samples per series for E[r_i]
TRAIN_FRACTION = 0.8      # must match training normalization split
MIN_FRESH_DAYS = 4        # max calendar days since last data point


def _fail(msg: str) -> None:
    print(f"[infer] FATAL: {msg}", file=sys.stderr)
    sys.exit(3)


def load_checkpoint(path: str, device: torch.device):
    from model import TACTiSModel  # repo root on sys.path

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        _fail(f"checkpoint {path} has no 'model' key (keys={list(ckpt.keys())})")
    state = ckpt["model"]
    # per-series embeddings pin num_series — infer from state dict
    num_series = state["flow_series_encoder.weight"].shape[0]
    print(f"[infer] checkpoint loaded: stage={ckpt.get('stage')} "
          f"epoch={ckpt.get('epoch')} num_series={num_series}")
    return state, num_series


def load_recent_data(csv_path: str):
    """Load merged all_stocks.csv, return (close_df, ticker_list)."""
    df = pd.read_csv(csv_path, usecols=["date", "ticker", "close"])
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="ticker", values="close")
    close = close.sort_index()
    return close


def check_freshness(close: pd.DataFrame) -> None:
    last = close.index.max()
    age = (pd.Timestamp.now(tz=None).normalize() - last).days
    print(f"[infer] data window {close.index.min().date()} .. {last.date()} "
          f"({len(close)} rows, age {age}d)")
    if age > MIN_FRESH_DAYS + 3:  # 3 extra days for long weekends
        _fail(f"stale data: last row {last.date()} is {age}d old")


def build_hist_tensor(close: pd.DataFrame, num_series: int,
                      series_limit: int | None):
    """Returns (hist_value[1,S,hist_len], norm_stats, tickers) on CPU."""
    # align columns to training universe
    tickers = list(close.columns)
    if num_series is not None and len(tickers) != num_series:
        if series_limit is None:
            _fail(f"universe mismatch: data has {len(tickers)} tickers, "
                  f"checkpoint expects {num_series}")
    if series_limit:
        tickers = tickers[:series_limit]
        close = close[tickers]
    returns = close.pct_change(fill_method=None).iloc[1:]  # arithmetic returns

    # normalize with train-region stats (mirrors CustomReturnDataset)
    train_end = int(len(returns) * TRAIN_FRACTION)
    mean = returns.iloc[:train_end].mean()
    std = returns.iloc[:train_end].std().replace(0, 1.0)
    norm = ((returns - mean) / std).to_numpy(dtype=np.float32)  # [T,S]

    # NaN policy: drop series with any NaN in the hist window, like training mask
    window = norm[-HIST_LEN:]                                  # [hist_len,S]
    valid = ~np.isnan(window).any(axis=0)
    if valid.sum() == 0:
        _fail("all series invalid in history window")
    if not valid.all():
        print(f"[infer] dropping {(~valid).sum()} series with NaN in window")
    window = window[:, valid]
    tickers = [t for t, v in zip(tickers, valid) if v]

    hist = torch.from_numpy(window.T[None, ...])               # [1,S,hist_len]
    return hist, tickers


def sample_expected_returns(model, hist: torch.Tensor, device, num_samples,
                            series_subset=None):
    from trainer import make_time_tensors

    hist = hist.to(device)
    if series_subset is not None:
        hist_time, pred_time = make_time_tensors(
            HIST_LEN, PRED_LEN, 1, device)
        samples = model.sample_subset(
            num_samples, hist_time, hist, pred_time, series_subset)
        # [1, S_sub, hist+pred, N] -> predicted window only
        pred = samples[:, :, HIST_LEN:, :]
    else:
        hist_time, pred_time = make_time_tensors(
            HIST_LEN, PRED_LEN, 1, device)
        samples = model.sample(num_samples, hist_time, hist, pred_time)
        pred = samples[:, :, HIST_LEN:, :]

    pred = pred.squeeze(0).cpu().numpy()   # [S, pred_len, N]
    # de-normalize per series: r_real = r_norm * std + mean (per series stats)
    return pred


def weights_from_samples(pred: np.ndarray, mean: np.ndarray, std: np.ndarray,
                         tickers: list[str], max_assets: int = 20):
    """E[r] over MC samples in RAW return space -> top-K proportional weights."""
    # pred: [S, pred_len, N] (S=series axis 0); mean/std: [S]
    raw = pred * std[:, None, None] + mean[:, None, None]
    horizon = np.prod(1.0 + raw, axis=1) - 1.0        # [S, N]
    exp_ret = horizon.mean(axis=0)                    # [S]
    # rank by expected return, keep top-K, proportional weights (long-only)
    order = np.argsort(-exp_ret)[:max_assets]
    w = np.clip(exp_ret[order], 0.0, None)
    if w.sum() <= 0:
        _fail("non-positive expected returns everywhere — refusing to emit weights")
    w = w / w.sum()
    return {tickers[i]: round(float(w[j]), 6)
            for j, i in enumerate(order) if i < len(tickers)}


def build_model_from_state(state: dict, num_data_series: int,
                           device: torch.device, num_ckpt_series: int):
    """Build a TACTiSModel from cfg/model/tactis_medium.yaml + ckpt state.

    Smoke subset mode (num_data_series < num_ckpt_series) rebinds the
    per-series embedding tables to the first ``num_data_series`` rows so the
    subset-conditioned sampler can look them up (mirrors subset_scope).
    """
    from model import TACTiSModel
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.environ.get("TACTIS_MODEL_CFG",
                          str(REPO / "cfg" / "model" / "tactis_medium.yaml")))
    params = OmegaConf.to_container(cfg, resolve=True)
    params.pop("name", None)
    params["num_series"] = num_data_series
    if params.get("bagging_size") and params["bagging_size"] > num_data_series:
        params["bagging_size"] = num_data_series  # subset smoke mode
    # stage-2 checkpoints carry copula params; the yaml has skip_copula=true
    # (stage-1 pretraining config) — flip it so the module tree matches
    if any(k.startswith("copula_encoder") for k in state):
        params["skip_copula"] = False
    model = TACTiSModel(params, device=device)

    if num_data_series == num_ckpt_series:
        model.load_state_dict(state, strict=True)
    else:
        # subset smoke: slice per-series tables, load the rest strictly
        sliced = {k: (v[:num_data_series] if v.shape[0] == num_ckpt_series else v)
                  for k, v in state.items()}
        model.load_state_dict(sliced, strict=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="all_stocks.csv (date,ticker,close)")
    ap.add_argument("--out", required=True, help="output weights JSON")
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--series-limit", type=int, default=None,
                    help="smoke mode: only first K series (subset sampling)")
    ap.add_argument("--max-assets", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--norm-stats", default=None,
                    help="norm_stats.json (training-consistent z-score stats, "
                         "from calc_norm.py). Strongly recommended: recompute-"
                         "from-data stats differ from training stats.")
    ap.add_argument("--skip-freshness", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    state, num_series = load_checkpoint(args.checkpoint, device)

    close = load_recent_data(args.data)
    if not args.skip_freshness:
        check_freshness(close)

    from infer import build_model_from_state  # same module (app dir on path)
    series_limit = args.series_limit
    if series_limit is not None:
        series_limit = min(series_limit, len(close.columns))
    model = build_model_from_state(state, len(close.columns)
                                   if series_limit is None
                                   else series_limit, device, num_series)
    model.eval()

    if args.norm_stats:
        ns = json.loads(Path(args.norm_stats).read_text())
        all_mean = np.array(ns["mean"], dtype=np.float64)
        all_std = np.array(ns["std"], dtype=np.float64)
        if len(all_mean) != num_series:
            _fail(f"norm_stats has {len(all_mean)} series, ckpt has {num_series}")
        col_index = {t: i for i, t in enumerate(ns["tickers"])}
        missing = [t for t in close.columns if t not in col_index]
        if missing:
            _fail(f"{len(missing)} data tickers missing from norm_stats, "
                  f"e.g. {missing[:5]}")
        mean = np.array([all_mean[col_index[t]] for t in close.columns])
        std = np.array([all_std[col_index[t]] for t in close.columns])
        print(f"[infer] using training-consistent norm_stats "
              f"(universe {len(all_mean)})")
    else:
        # legacy fallback: recompute from data (WRONG vs training stats unless
        # the data window covers the full training period)
        returns = close.pct_change(fill_method=None).iloc[1:]
        train_end = int(len(returns) * TRAIN_FRACTION)
        mean = returns.iloc[:train_end].mean().to_numpy()
        std = returns.iloc[:train_end].std().replace(0, 1.0).to_numpy()
        print("[infer] WARNING: recomputing norm stats from data — "
              "pass --norm-stats for training-consistent stats")

    hist, tickers = build_hist_tensor(close, num_series, args.series_limit)
    print(f"[infer] sampling {args.num_samples} paths for {len(tickers)} series")

    with torch.no_grad():
        if args.series_limit:
            subset = list(range(len(tickers)))
            pred = sample_expected_returns(model, hist, device,
                                           args.num_samples, subset)
        else:
            pred = sample_expected_returns(model, hist, device, args.num_samples)

    # per-series stats aligned to kept tickers
    idx = [list(close.columns).index(t) for t in tickers]
    m = mean[idx]
    s = std[idx]
    weights = weights_from_samples(pred, m, s, tickers, args.max_assets)

    out = Path(args.out)
    out.write_text(json.dumps(weights, indent=2))
    print(f"[infer] wrote {len(weights)} positions to {out}")
    print(json.dumps(dict(list(weights.items())[:5]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
