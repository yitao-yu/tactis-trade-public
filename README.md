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

All tables below: fixed code (closed-market tradability mask + account gross/net caps),
US shortable / CN long-only.

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
| Final NAV | 6.350 (+535 %) | 5.992 (+499 %) |
| GMRR | 0.0384 %/day (ann 10.2 %) | 0.0372 %/day (ann 9.8 %) |
| Daily Sharpe | 0.0504 | 0.0434 |
| Max DD | −28.5 % | −26.8 % |
| Turnover/day | 0.191 | 0.155 |
| Costs (NAV) | 0.669 | 1.450 |
| Ann vol | 12.1 % | 13.6 % |
| Beta / β-share | 0.53 / 97 % | 0.56 / 107 % |
| Alpha (ann.) | +0.28 % | −0.65 % |
| Cash mean | 42.5 % | 44.1 % |

**per-year breakdown (total return / GMRR / Sharpe / maxDD)**

| Year | 3 bp ret | 3 bp GMRR | 3 bp Shp | 3 bp DD | 3 bp cash | 10 bp ret | 10 bp GMRR | 10 bp Shp | 10 bp DD | 10 bp cash |
|---|---|---|---|---|---|---|---|---|---|---|
| 2008 | −21.3 % | −0.100 % | −0.102 | −28.5 % | 58.5 % | −20.3 % | −0.095 % | −0.097 | −26.8 % | 60.9 % |
| 2009 | +46.1 % | 0.147 % | 0.188 | −6.4 % | 52.2 % | +38.1 % | 0.125 % | 0.169 | −6.7 % | 57.1 % |
| 2010 | +7.0 % | 0.026 % | 0.046 | −9.9 % | 53.1 % | +8.3 % | 0.031 % | 0.052 | −10.2 % | 56.5 % |
| 2011 | −6.4 % | −0.025 % | −0.044 | −10.4 % | 49.5 % | −9.4 % | −0.038 % | −0.066 | −12.8 % | 55.4 % |
| 2012 | +7.1 % | 0.027 % | 0.043 | −8.5 % | 45.9 % | +3.8 % | 0.014 % | 0.024 | −11.3 % | 53.2 % |
| 2013 | +15.2 % | 0.055 % | 0.104 | −6.5 % | 47.9 % | +13.7 % | 0.050 % | 0.087 | −6.3 % | 50.8 % |
| 2014 | +16.3 % | 0.058 % | 0.117 | −5.9 % | 42.3 % | +15.8 % | 0.057 % | 0.107 | −5.6 % | 48.9 % |
| 2015 | +29.8 % | 0.101 % | 0.079 | −21.4 % | 40.3 % | +31.3 % | 0.105 % | 0.069 | −26.0 % | 40.6 % |
| 2016 | +3.6 % | 0.014 % | 0.017 | −11.6 % | 45.0 % | +8.3 % | 0.031 % | 0.038 | −11.0 % | 48.9 % |
| 2017 | +10.2 % | 0.038 % | 0.101 | −3.9 % | 42.1 % | +7.6 % | 0.029 % | 0.065 | −6.5 % | 48.6 % |
| 2018 | −9.3 % | −0.038 % | −0.055 | −12.9 % | 40.0 % | −13.9 % | −0.058 % | −0.079 | −15.8 % | 45.1 % |
| 2019 | +31.0 % | 0.104 % | 0.150 | −8.4 % | 37.2 % | +28.6 % | 0.097 % | 0.134 | −10.2 % | 45.3 % |
| 2020 | +29.1 % | 0.098 % | 0.117 | −10.7 % | 39.6 % | +34.2 % | 0.113 % | 0.127 | −9.3 % | 46.7 % |
| 2021 | +21.2 % | 0.075 % | 0.122 | −6.2 % | 32.6 % | +20.1 % | 0.071 % | 0.101 | −7.8 % | 39.7 % |
| 2022 | −11.4 % | −0.047 % | −0.059 | −19.9 % | 31.5 % | −14.4 % | −0.060 % | −0.069 | −20.1 % | 37.2 % |
| 2023 | +4.7 % | 0.018 % | 0.029 | −10.9 % | 32.5 % | +1.3 % | 0.005 % | 0.007 | −12.2 % | 34.1 % |
| 2024 | +6.6 % | 0.025 % | 0.025 | −9.7 % | 34.3 % | +10.6 % | 0.039 % | 0.034 | −11.2 % | 24.2 % |
| 2025 | +26.5 % | 0.091 % | 0.114 | −10.1 % | 39.9 % | +24.9 % | 0.086 % | 0.090 | −12.0 % | 19.1 % |
| 2026\* | +14.3 % | 0.087 % | 0.096 | −7.9 % | 45.5 % | +26.3 % | 0.151 % | 0.097 | −20.4 % | 14.1 % |

**Loss-violation (account-level — the roll-based version is unreliable)**
| | daily <1 % / <3 % | 4-day span <1 % / <3 % / <6 % |
|---|---|---|
| 3 bp | 7.2 % / 0.4 % | 18.5 % / 2.97 % / 0.35 % |
| 10 bp | 8.8 % / 0.6 % | 20.6 % / 3.79 % / 0.44 % |

> ⚠️ The **2008-2026** tables above are the **base** checkpoint on the fixed code.
> Note `base@10 bp` leaked leverage (`over_gross_days = 646`, max gross 1.34) because the account gross cap is approximate — treat its total return with caution; `base@3 bp` is clean.

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
