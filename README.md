**AGENTIC USERS, ROBOTS, OR HUMAN WHO NEED MORE DETAILS**: Please refer to `AGENTS.md` (and make changes to it to suit your environment)! This README will be written by LLM.  


## Pre-training

```
# 64
python train.py training.epochs_stage1=64 training.epochs_stage2=64 wandb.enabled=true
# or 128
python train.py training.epochs_stage1=128 training.epochs_stage2=128 wandb.enabled=true
```

## RL Post-training

```
python post_training.py rl.resume_ckpt=<ckpt>
```

## Venue 0 backtest — run commands

**Mock smoke (CPU, no GPU/model) — quick correctness check:**
```bash
python backtest.py backtest.mock_model=true \
  backtest.start_date=2025-09-20 backtest.end_date=2025-10-15 \
  backtest.n_samples=4 training.device=cpu
```

**Full range (2008-01-02 → 2026-08-10), RL 09-10 checkpoint, both markets at a fee:**
```bash
CKPT=outputs/2026-09-10/17-48-39-523325/checkpoints/rl_step003504_sharpe0.0478.pt

# 3 bp
python backtest.py \
  backtest.checkpoint=$CKPT backtest.output_dir=outputs/venv0_full_rl0910_3bp \
  backtest.start_date=2008-01-02 backtest.end_date=2026-08-10 \
  backtest.commission_model=rate backtest.us_commission_bps=3 backtest.cn_commission_bps=3 \
  backtest.allocator.commission_rate=0.0003

# 10 bp  (us/cn = 10, commission_rate = 0.001)
# 20 bp  (us/cn = 20, commission_rate = 0.002)
```

**Behavior guarantees in the current code (2026-09-18 fixes):**
- tradability mask — only names with a finite close today **and** finite open tomorrow are
  allocated; closed-market (CN/US holiday) holdings are **carried, not sold**;
- account caps — `gross ≤ 1` and `net ≤ 1` (no borrowing); the executor also skips any
  buy that would drive cash below zero;
- turnover penalty = `turnover_coef × max(commission_rate, 3e-4)` (default `16.7 × 3e-4 ≈ 0.005`);
- diagnostics: `nav.csv` has `cash_weight` (= account cash = 1 − net), `cash_unengaged`
  (= 1 − gross), `partition_tile_gap`; `backtest_metrics.json` has `neg_cash_days`,
  `over_gross_days`, `max_gross_exposure`, `max_abs_tile_gap`.

## Backtest Results

#### Test range 2025-2026.8.10(415 days)

**0 bp**
| Checkpoint | NAV | Total | GMRR/day | Sharpe | Max DD | Turnover | Costs | Vol | Cash (mean/min/max) | Beta | Alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.523 | +52.3 % | 0.1016 % | 0.1241 | −10.4 % | 0.170 | 0.000 | 13.0 % | 42 / 22 / 100 % | 0.76 | +8.3 % |
| rl0910 | 1.288 | +28.8 % | 0.0612 % | 0.0842 | −12.3 % | 0.144 | 0.000 | 11.5 % | 36 / 13 / 100 % | 0.69 | −0.1 % |
| rl0914 | 1.279 | +27.9 % | 0.0595 % | 0.0745 | −12.6 % | 0.171 | 0.000 | 12.7 % | 29 / 5 / 100 % | 0.83 | −3.8 % |

**3 bp**
| Checkpoint | NAV | Total | GMRR/day | Sharpe | Max DD | Turnover | Costs | Vol | Cash (mean/min/max) | Beta | Alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.480 | +48.0 % | 0.0948 % | 0.1154 | −10.6 % | 0.172 | 0.027 | 13.0 % | 42 / 23 / 100 % | 0.76 | +6.6 % |
| rl0910 | 1.270 | +27.0 % | 0.0577 % | 0.0797 | −12.4 % | 0.144 | 0.021 | 11.5 % | 36 / 13 / 100 % | 0.69 | −1.0 % |
| rl0914 | 1.260 | +26.0 % | 0.0558 % | 0.0694 | −12.7 % | 0.170 | 0.025 | 12.8 % | 29 / 6 / 100 % | 0.84 | −4.9 % |

**10 bp**
| Checkpoint | NAV | Total | GMRR/day | Sharpe | Max DD | Turnover | Costs | Vol | Cash (mean/min/max) | Beta | Alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.575 | +57.5 % | 0.1098 % | 0.1070 | −12.2 % | 0.145 | 0.076 | 16.3 % | 41 / 21 / 100 % | 0.92 | +6.8 % |
| rl0910 | 1.276 | +27.6 % | 0.0589 % | 0.0668 | −12.4 % | 0.142 | 0.066 | 14.0 % | 45 / 13 / 100 % | 0.74 | −1.8 % |
| rl0914 | 1.373 | +37.3 % | 0.0766 % | 0.0835 | −12.3 % | 0.133 | 0.067 | 14.6 % | 30 / 3 / 100 % | 0.91 | −1.4 % |

**20 bp**
| Checkpoint | NAV | Total | GMRR/day | Sharpe | Max DD | Turnover | Costs | Vol | Cash (mean/min/max) | Beta | Alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.593 | +59.3 % | 0.1126 % | 0.0838 | −22.8 % | 0.087 | 0.085 | 21.3 % | 35 / 16 / 100 % | 1.12 | +3.0 % |
| rl0910 | 1.279 | +27.9 % | 0.0594 % | 0.0617 | −13.0 % | 0.114 | 0.106 | 15.3 % | 52 / 23 / 112 % | 0.75 | −1.9 % |
| rl0914 | 1.315 | +31.5 % | 0.0661 % | 0.0742 | −11.5 % | 0.116 | 0.115 | 14.1 % | 40 / 2 / 100 % | 0.85 | −2.7 % |

#### Test range 2008-2026.8.10, training and val included

**Overall**
| | 3 bp | 10 bp |
|---|---|---|
| Final NAV | 3.724 (+272 %) | 2.297 (+130 %) |
| GMRR | 0.0273 %/day (ann 7.1 %) | 0.0173 %/day (ann 4.5 %) |
| Daily Sharpe | 0.0364 | 0.0218 |
| Max DD | −24.5 % | −28.0 % |
| Turnover/day | 0.350 | 0.179 |
| Costs (NAV) | 0.951 | 1.140 |
| Ann vol | 11.9 % | 12.6 % |
| Beta / β-share | 0.35 / 91 % | 0.32 / 131 % |
| Alpha (ann.) | +0.63 % | −1.35 % |
| Cash mean | 33.9 % | 49.1 % |

**per-year breakdown (total return / GMRR / Sharpe / maxDD)**

| Year | 3 bp ret | 3 bp GMRR | 3 bp Shp | 3 bp DD | 3 bp cash | 10 bp ret | 10 bp GMRR | 10 bp Shp | 10 bp DD | 10 bp cash |
|---|---|---|---|---|---|---|---|---|---|---|
| 2008 | −15.0 % | −0.068 % | −0.063 | −24.5 % | 78.8 % | −18.0 % | −0.083 % | −0.072 | −27.4 % | 90.8 % |
| 2009 | +9.7 % | 0.036 % | 0.037 | −18.3 % | 72.1 % | +0.4 % | 0.002 % | 0.001 | −22.2 % | 89.0 % |
| 2010 | +19.0 % | 0.067 % | 0.099 | −10.8 % | 32.3 % | +15.7 % | 0.057 % | 0.090 | −10.1 % | 46.1 % |
| 2011 | −14.9 % | −0.062 % | −0.102 | −18.7 % | 35.1 % | −16.9 % | −0.071 % | −0.134 | −19.8 % | 50.7 % |
| 2012 | +4.9 % | 0.018 % | 0.028 | −12.3 % | 25.3 % | +2.6 % | 0.010 % | 0.017 | −12.6 % | 43.2 % |
| 2013 | +18.5 % | 0.066 % | 0.114 | −7.7 % | 23.2 % | +17.0 % | 0.061 % | 0.113 | −7.0 % | 41.4 % |
| 2014 | +17.6 % | 0.063 % | 0.119 | −4.9 % | 18.7 % | +11.8 % | 0.043 % | 0.090 | −5.8 % | 39.0 % |
| 2015 | +29.2 % | 0.099 % | 0.097 | −19.7 % | 19.1 % | +24.0 % | 0.083 % | 0.063 | −27.0 % | 30.6 % |
| 2016 | +4.6 % | 0.017 % | 0.023 | −9.2 % | 21.9 % | +2.5 % | 0.010 % | 0.014 | −10.4 % | 42.0 % |
| 2017 | +11.2 % | 0.041 % | 0.100 | −4.3 % | 19.7 % | +7.4 % | 0.028 % | 0.069 | −5.4 % | 41.3 % |
| 2018 | −16.4 % | −0.070 % | −0.095 | −19.7 % | 23.4 % | −16.7 % | −0.071 % | −0.108 | −18.3 % | 41.6 % |
| 2019 | +29.2 % | 0.099 % | 0.140 | −7.4 % | 20.4 % | +24.3 % | 0.084 % | 0.129 | −8.5 % | 41.0 % |
| 2020 | +12.9 % | 0.047 % | 0.047 | −12.8 % | 51.6 % | +13.1 % | 0.047 % | 0.046 | −13.7 % | 68.8 % |
| 2021 | +19.6 % | 0.070 % | 0.126 | −4.0 % | 20.5 % | +18.9 % | 0.067 % | 0.110 | −5.6 % | 39.9 % |
| 2022 | −10.2 % | −0.042 % | −0.055 | −19.2 % | 34.1 % | −13.5 % | −0.056 % | −0.076 | −19.6 % | 44.3 % |
| 2023 | +2.4 % | 0.009 % | 0.017 | −11.4 % | 26.8 % | −1.4 % | −0.006 % | −0.010 | −12.7 % | 39.5 % |
| 2024 | +10.0 % | 0.037 % | 0.041 | −8.8 % | 34.9 % | +5.8 % | 0.022 % | 0.022 | −9.6 % | 45.3 % |
| 2025 | +13.4 % | 0.049 % | 0.074 | −10.8 % | 41.4 % | +14.0 % | 0.051 % | 0.068 | −12.1 % | 50.2 % |
| 2026\* | +9.2 % | 0.057 % | 0.086 | −5.6 % | 56.2 % | +12.1 % | 0.074 % | 0.082 | −9.1 % | 51.9 % |

**Loss-violation (account-level — the roll-based version is unreliable)**
| | daily <1 % / <3 % | 4-day span <1 % / <3 % / <6 % |
|---|---|---|
| 3 bp | 7.1 % / 0.4 % | 20.0 % / 2.83 % / 0.40 % |
| 10 bp | 7.5 % / 0.6 % | 20.4 % / 3.29 % / 0.37 % |

> ⚠️ The tables above were produced **before** the 2026-09-18 fixes (closed-market
> tradability mask, account gross/net caps, hold-frozen-positions, stable reconcile).
> Re-run the commands below to refresh them.

---

## Live venues — IBKR (Venue 1) + Xueqiu (Venue 2)

Executors and safety guards are implemented; **dry-run is the default everywhere**
and a live runner (`run_live.py`) is deferred. Config lives in `cfg/application.yaml`
(separate from `cfg/config.yaml`).

- **Venue 1 — IBKR** (`trading/venues/ibkr.py`): US (long+short where shortable) and
  China A-shares via Stock Connect (long-only, 100-share lots), `ib_async`,
  at-the-open (`MOO`/`OPG`) orders, account `sync()`.
- **Venue 2 — Xueqiu** (`trading/venues/xueqiu.py`): CN long-only, ≤30 holdings,
  ≥1 % each, target-based full rebalance via `easytrader` (money-market cash proxy).

Live market data is **not** pulled from the trading APIs yet — the model reads the
offline `all_stocks.csv` (yfinance/baostock); broker APIs supply account/order state only.

**Safety gate** (`trading/live.py`): an order is submitted only if the venue block's
`live: true` **and** env `TACTIS_LIVE=1`; otherwise dry-run. Kill-switch
(`TACTIS_TRADING_KILL=1` or the `live.kill_switch_file` sentinel) and `max_notional`
apply on top.

```bash
# install the application-only dependencies (live machines only)
pip install -r requirements-app.txt

# offline tests for the live layer — no broker, no network, no cookie
pytest tests/test_live_guards.py tests/test_ibkr_venue.py tests/test_xueqiu_venue.py -v

# full suite
pytest tests/
```

Enable live (later, on EC2): set the venue's `live: true` and `dry_run: false` in
`cfg/application.yaml`, then export `TACTIS_LIVE=1` (and *not* `TACTIS_TRADING_KILL=1`).
