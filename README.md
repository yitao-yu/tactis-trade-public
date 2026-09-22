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

Test range 2025-2026.8.10(415 days)

**0 bp — ⚠️ penalty-INACTIVE (pre-fix; turnover ≈ 0.38)**
| Checkpoint | NAV | Total | GMRR/day | Sharpe | Max DD | Turnover | Costs | Vol | Cash (mean/min/max) | Beta | Alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.271 | +27.1 % | 0.0579 % | 0.0867 | −10.9 % | 0.380 | 0 | 10.6 % | 49 / −16 / 124 % | 0.60 | +1.0 % |
| rl0910 | 1.246 | +24.6 % | 0.0532 % | 0.0935 | −9.6 % | 0.402 | 0 | 9.0 % | 46 / 13 / 104 % | 0.49 | +2.4 % |
| rl0914 | 1.224 | +22.4 % | 0.0488 % | 0.0826 | −8.3 % | 0.363 | 0 | 9.4 % | 44 / 23 / 100 % | 0.57 | −0.5 % |

**3 bp — ⚠️ penalty-INACTIVE (pre-fix; turnover ≈ 0.38)**
| Checkpoint | NAV | Total | GMRR/day | Sharpe | Max DD | Turnover | Costs | Vol | Cash (mean/min/max) | Beta | Alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.212 | +21.2 % | 0.0465 % | 0.0696 | −11.2 % | 0.380 | 0.051 | 10.6 % | 49 / −18 / 122 % | 0.60 | −1.9 % |
| rl0910 | 1.185 | +18.5 % | 0.0411 % | 0.0722 | −9.8 % | 0.402 | 0.054 | 9.0 % | 46 / 13 / 104 % | 0.49 | −0.7 % |
| rl0914 | 1.169 | +16.9 % | 0.0378 % | 0.0639 | −8.8 % | 0.363 | 0.050 | 9.4 % | 44 / 23 / 100 % | 0.57 | −3.3 % |

**20 bp — ✅ penalty-ACTIVE (post-fix; turnover ≈ 0.12)**

| Checkpoint | NAV | Total | GMRR/day | Sharpe | Max DD | Turnover | Costs | Vol | Cash (mean/min/max) | Beta | Alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.527 | +52.7 % | 0.1023 % | 0.0987 | −11.9 % | 0.118 | 0.121 | 16.4 % | 45 / 17 / 112 % | 0.87 | +6.1 % |
| rl0910 | 1.248 | +24.8 % | 0.0536 % | 0.0591 | −14.7 % | 0.120 | 0.108 | 14.4 % | 58 / 31 / 190 % | 0.66 | −1.5 % |
| rl0914 | 1.282 | +28.2 % | 0.0600 % | 0.0709 | −10.8 % | 0.112 | 0.109 | 13.4 % | 47 / 24 / 100 % | 0.77 | −2.2 % |

**Overall(all dates 2008.9-2026.8.10, training and val included)**
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
