# AGENTS.md — tactis-trade

## Project overview

Time-series forecasting for financial trading using the [TACTiS](https://github.com/Servicenow/tactis) transformer-attentional copulas model. Python 3.10, PyTorch, no formal build system or test framework (early-stage prototype).

## Setup

The local environment is `tactis` (Python 3.10, PyTorch 2.10, tactis 0.1.3, CUDA available). Activate it:

```bash
conda activate tactis
```

Or if this is a session from wsl: 

```bash
conda activate tactis-trade
```

To recreate from scratch:

```bash
conda create --name tactis-trade python=3.10 --yes
conda activate tactis-trade
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install tactis
```

For agents/users from China, use the mirrors for higher speed: 

```bash
conda create --name tactis-trade python=3.10 --yes
conda activate tactis-trade
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch -f https://mirrors.aliyun.com/pytorch-wheels/cu118 && pip install -r requirements.txt 
pip install -r requirements.txt
pip install tactis
```

## Training

Two-stage training with Hydra config:

```bash
python train.py                                              # custom close returns + tactis_medium (default)
python train.py dataset=mitsui                               # mitsui dataset
python train.py dataset=custom training.epochs_stage1=100    # override params
```

Key training config options (`training.`):
- `epochs_stage1`, `epochs_stage2` — epochs per stage
- `batch_size`, `lr_stage1`, `lr_stage2`, `weight_decay`
- `clip_grad` — gradient clipping threshold
- `val_every_epochs` — validation frequency
- `topk_checkpoints` — number of best-val checkpoints to keep (default 10)

Reproducibility: seed is applied before model/dataset creation (`torch.manual_seed`, `np.random.seed`, `random.seed`, `cudnn.deterministic`).

Configs live in `cfg/`:
- `cfg/config.yaml` — root defaults (dataset, model, training)
- `cfg/dataset/` — custom.yaml, mitsui.yaml
- `cfg/model/` — tactis_medium.yaml

### Wandb (optional)

Weights & Biases logging is available via `cfg.wandb`:

```bash
python train.py wandb.enabled=true                            # offline mode by default
python train.py wandb.enabled=true wandb.mode=online          # sync to cloud
python train.py wandb.enabled=true wandb.project=my-project   # custom project
```

The default `wandb.mode: offline` writes metrics locally to `./wandb/` for development — no network required.  Set `wandb.mode: online` for cloud sync.

Logged per-epoch: `stage{1,2}/train_loss`, `train_marginal`, `train_copula`, `val_marginal`, `val_copula`, `val_loss`, `epoch` plus per-stage validation metrics (see below).  Hyperparameters are auto-logged from the Hydra config.

### Validation metrics

When `training.metrics.enabled: true` (default), `compute_validation_metrics` draws MC samples
from the model on the validation set every epoch and logs:

- **recall@k**: `stage{s}/val/recall@k_mean`, `stage{s}/val/recall@k_q95` — whether the true
  top-1 return asset is in the predicted top-k (ranked by sample mean or 95th percentile).
- **Portfolio**: per-allocator `stage{s}/val/portfolio/{name}/return`, `var`, `cvar`,
  `mean_return`, `sharpe`.  All returns are **raw daily arithmetic returns** (e.g. 0.0023 =
  0.23 %), de-normalized from z-score space using per-series ``norm_stats`` from the
  training split.

Allocators available:

| Allocator | Logic |
|---|---|
| `proportional` | `w_i ∝ max(0, E[r_i])` |
| `var_aware` | Filter by `VaR_α > threshold`, then proportional |
| `var_optimizer` | Gradient-descent: max mean s.t. CVaR soft constraint + hard constraint projection |
| `mv_neutral_cvar` | Gradient-descent: max ``μᵀw - (λ/2)Var(w·r) - γ·CVaR``, dollar-neutral (``1ᵀw=0``), position limits, optional beta neutrality |
| `mv_long` | Gradient-descent long-only: max ``μᵀw - (λ/2)Var(w·r) - γ·CVaR``, softmax over logits, optional beta neutrality |
| `var_deployment` | Long-short MV: max ``μᵀw - (λ/2)Var(w·r)`` with soft beta-neutrality + soft CVaR floor/ceiling + hard central return band that de-leverages into cash.  ``use_hard_band=false`` skips the band and normalises gross to 1 (full investment, no cash) |

All allocators live in `portfolio/` (importable package), share the same constraint projection
pipeline (`max_assets`, `max_weight`, `allow_short`, `max_gross_exposure`), and are configurable
via `training.metrics.allocators`.

The constraint pipeline runs post-construction for every allocator:
1. Zero invalid positions.
2. **Cardinality** — truncate to top-K by |weight| magnitude (preserves the allocator's actual
   allocation, not a heuristic score).
3. Per-asset cap (cash exempt).
4. Long-only clamp.
5. Gross-exposure cap.
6. Normalise to sum=1.

A cash asset (zero return, zero variance) is appended before allocation and is always preserved
through cardinality and exempt from max_weight — the optimizer can freely allocate to cash when
it reduces CVaR violation.

Constraints incompatible with dollar-neutral (cardinality, gross-exposure cap, normalisation)
are skipped during projection.  Beta data pre-computed offline against SPY/510050 market proxies.

**Memory note**: ``model.sample()`` operates on all series without bagging, so attention
scales as ``O((series × timesteps)²)``.  For the custom dataset (1,302 series) this exceeds GPU
limits at batch_size > 1.  ``compute_validation_metrics`` works around this by calling
``model.sample()`` in micro-batches — correlation is fully preserved, only the batch
dimension is chunked.  Mitsui (143 series) fits in a single call.
Tune ``sample_micro_batch`` (default 4) to balance GPU RAM and speed.

Key config options (`training.metrics`):
- `enabled: true` — toggle metrics on/off
- `n_samples: 10` — MC samples per validation window
- `recall_k: 10`, `var_alpha: 0.1`
- `sample_micro_batch: 8` — batch items per ``model.sample()`` call
- `allocators` — list of ``{name, constraints, ...}`` dicts

### Config rationale

| Default | Reason |
|---|---|
| `n_samples: 10` | Tradeoff between CVaR noise and sampling speed.  Increase once copula validation is viable. |
| `sample_micro_batch: 8` | ~4× faster than batch=1 while staying within 8 GB GPU for 1,302 series. |
| `var_alpha: 0.1` | 90 % confidence for VaR/CVaR.  At n_samples=10 the 5 % tail would contain only 1 sample. |
| `epochs_stage1: 30` | Val loss plateaus around epoch 20–25 for the custom dataset.  128 epochs showed no marginal recall/portfolio improvement. |
| `epochs_stage2: 30` | Copula training converges fast (2–3 epochs show meaningful improvement).  Wall time is dominated by validation sampling, not training steps. |
| `var_optimizer.max_loss: 0.01` | Soft CVaR constraint: tail-mean return must stay above -1 % daily.  Tight but achievable with cash allocation. |

### Experimental observations

| Finding | Detail |
|---|---|
| **q95 beats mean for recall** | ``recall@10_q95`` consistently outperforms ``recall@10_mean`` (e.g. 9.2 % vs 4.8 % in raw-return space).  The distribution's upper tail contains more signal than the point estimate — validates the probabilistic approach. |
| **Raw-return recall is hard** | At 1,302 stocks, cross-sectional return ranking is dominated by volatility noise.  Random baseline for recall@10 is 0.8 % (= 10 / 1,302).  The model achieves 6–11× random in raw space.  Z-score space gives ~60 % recall for the same task — the model's signal lives in standardized-return space. |
| **Stage 2 copula bottleneck** | ``model.sample()`` with copula is ``O(S²)`` autoregressively.  1 epoch of stage 2 validation takes ~10–15 min with ``sample_micro_batch=8, n_samples=10``.  Training steps are fast (bagging_size=100), validation is the wall-time bottleneck. |
| **Marginal appears to drift in stage 2** | ``marginal_logdet`` values shift by ~0.01 between epochs — this is random noise from bagging different series subsets, not parameter updates.  Flow parameters have ``requires_grad=False`` during stage 2. |
| **Proportional vs var_optimizer** | Early stage 1: proportional achieves higher Sharpe than var_optimizer (0.03 vs 0.008).  The optimizer likely needs more MC samples for reliable CVaR estimation; 1-sample tails at n_samples=10 are noisy. |
| **Var/CVaR degenerate at n_samples=10** | At ``var_alpha=0.05`` and ``n_samples=10``, ``tail_size = max(1, 0.5) = 1`` — VaR and CVaR are both the single worst draw.  The CVaR penalty is essentially picking one random sample as the tail estimate, making the constraint meaningless.  Both ``var_optimizer`` and ``var_aware`` are affected. |
| **soft_cvar_weight sweep — hypothesis 1 rejected** | Swept ``λ ∈ [0, 1000]`` on fixed MC samples for both val and test.  Val: mean_return negative at all λ (best −0.026 % at λ=1000).  Test: mean_return weakly positive at all λ (+0.14–0.15 %).  **No λ can flip val positive** — the model's predicted mean returns are slightly negative on val.  **val-vs-test sign flip** confirms the optimizer overfits to noise, not signal.  Tuning the CVaR penalty weight does not fix the underlying problem. |
| **Val vs test recall holds** | ``recall@10_q95`` = 10.1 % (val) vs 11.2 % (test) — the model's ranking ability generalises.  ``recall@10_mean`` = 6.2 % (val) vs 5.9 % (test).  The walk-forward test split shows no degradation in ranking quality. |
| **Portfolio returns ~zero on test** | Despite decent recall (11.2 % recall@10_q95 on test), both proportional and var_optimizer produce near-zero portfolio returns (0.07–0.12 % daily, sharpe ≈ 0).  The model ranks well but the return-level signal is too weak relative to cross-sectional volatility for any current allocator to convert into positive portfolio returns. |
| **MV neutral/long allocators — hypothesis 2** | Added ``mv_neutral_cvar`` (dollar-neutral, MV + CVaR) and ``mv_long`` (long-only, MV + CVaR).  Test results: proportional +0.067 % / sharpe 0.036, var_optimizer +0.109 % / 0.029, ``mv_neutral_cvar`` **−0.018 %** / −0.021, ``mv_long`` +0.051 % / 0.015.  ``mv_neutral_cvar`` produced gross_exposure=8.1 (massive leverage) with dollar_balance=0.016 (broken neutrality from clamp-zerocenter ordering).  ``mv_long`` was the most diversified (eff_n=40.4, max_assets=50) but lowest sharpe.  **No new allocator improved on the baseline.** |
| **RL embedding-only training degrades recall** | First RL run (train embeddings only, ``freeze_embeddings=false``, ``freeze_encoder=true``, ``freeze_decoder=true``): after 100 steps, recall@10_q95 dropped from 10.5 % → 0.0 % and mean_return collapsed to ~0.  The anchor weight (0.01) was too weak to prevent the REINFORCE gradient from destroying the model's predictive structure.  Sharps remained in the same ballpark (0.056 vs 0.065 for proportional) — likely a degenerate portfolio near zero variance. |
| **RL post-training helps risk but degrades recall (degenerates past ~step 300)** | Two 08-15 → RL runs verified (``reward=sharpe``, ``allocator=var_optimizer``, ``kl_beta=0.1``; ``reward_scale`` is irrelevant for ``sharpe``).  Post-training *helps*: var_optimizer var_violation drops from ~0.65 (base, per checkpoint filenames) to 0.34–0.56, and test sharpe roughly doubles at good checkpoints (var_optimizer 0.029 → 0.049; proportional 0.036 → 0.065).  It *costs* recall: ``recall@10_q95`` falls from ~10–11 % (base) to 5–10 %.  **08-19** (resume epoch 125): var_violation 0.56 (step 200) → 0.34 (step 300), recall@10_q95 8.5 % → 5.5 % (val) / 7.2 % → 5.2 % (test), then degrades to 0.93 by step 400.  **08-20** (resume epoch 103): best recall 9.1 % val / 10.4 % test at step 100 but var_violation stayed ~0.70–0.72; by step 300 val sharpe collapsed (~0.003) while test held up (0.049).  Continued training reliably diverges past the good region. |
| **Dollar-neutral collapses post-RL; market-neutral controls tail** | Across the 08-19/08-20 evals, ``mv_neutral_cvar`` (dollar-neutral) shrinks toward ~100 % cash: effective_n ~90–99 M, gross_exposure ~0.0004–0.005, return ~1e-6, sharpe ≈ 0 or negative.  Dollar neutrality is a poor fit — the model's edge is directional (cross-sectional ranking), not long-horizon mean reversion.  ``mv_long`` (market-beta penalty + MV + CVaR, long-only) instead posts the lowest var_violation at good checkpoints (0.03–0.09 vs var_optimizer 0.31–0.63) with positive sharpe (0.01–0.05). |
| **`var_deployment` — tail control works, but reward is degenerate** | 08-22 run (``reward=sharpe``, reward allocator = ``var_deployment``, ``kl_beta=0.1``).  On val, ``var_deployment`` holds var_violation at **1.9–2.1 %** through steps 100–200 (vs var_optimizer 29–99 % / proportional 50–99 %) but mean_return is ~0 (de-leveraging into cash); its sharpe (0.036 at step 200) is a tiny-variance artifact, not edge.  vv jumps to 41–45 % and sharpe goes negative (−0.014) by step 300–400.  recall@10_q95 collapses 9.7 % → 2.4 % between step 100 and 200 (the usual embedding drift).  The saved best-sharpe checkpoint (step 400, named by var_optimizer sharpe 0.0373) was the *worst* state for var_deployment — motivating checkpoint selection by var_deployment sharpe instead. |
| **Frozen-reference baseline + symmetric anchor fix flattening but enable over-concentration** | 08-24 run (``reward=custom_reward_product``, reward allocator = ``var_deployment`` ``use_hard_band=false``, ``kl_beta=0.1``, per-epoch validation + baseline refresh).  logp went −73 → +46 (crossed zero ~step 400) and ``sq`` (distance to reference) grew to 27 — the mirror image of the old flattening.  Cause: advantage turned positive (mean +0.038, 61 % positive) because the reference never refreshed — val sharpe only improved 0.0046→0.0131, never clearing the 0.02 refresh margin — so REINFORCE kept pushing logp up.  Remedy: smaller ``baseline_refresh_margin`` (→0.005) and a stronger ``kl_beta`` trust region. |
| **Reward improves only on the reward allocator; recall + long-only collapse** | 08-24: ``rl/reward`` −0.04 → +0.037 and reward-allocator ``rl/var_violation`` 0.73 → 0.47, but val recall@10_q95 8.4 % → 3.6 %, proportional sharpe 0.020 → 0.009, var_optimizer/mv_long → negative.  Optimizing the long-short reward distorts the predictive distribution away from the cross-sectional signal — the same RL-destroys-recall pattern, now via concentration instead of flattening. |
| **Optimism inflation — predicted tail rises while realized returns fall** | 08-24: for ``var_optimizer`` the predicted ``var`` (single min sample — ``var == cvar``, degenerate tail at n_samples=10) rose 0.007 → 0.018 while realized ``mean_return`` → −0.00012 and ``var_violation`` → 0.79 (mv_long: 0.0017 → 0.0050, vv → 0.63).  The model concentrates its distribution around a positive predicted mean, so its predicted worst-case rises but reality doesn't follow — the concrete signature of the recall collapse. |
| **Long-short is structurally harder than long-only — net exposure, not direction** | NLL training is direction-symmetric; the asymmetry is in the payoff.  A long-only book (net exposure = 1) captures market drift; a beta-neutral long-short book (net ≈ 0, since ``βᵀw ≈ Σw`` against an equal-weighted market proxy) earns only near-zero alpha.  At step 0, ``mv_long`` (long-only, same ``beta_penalty_weight=0.1``) got sharpe +0.018 vs var_deployment +0.005 — the beta penalty can only tilt a long book, not zero its net exposure.  Any CVaR optimizer (var_optimizer +0.002, var_deployment +0.005) is also noisy at n_samples=10 (tail_size=1). |
| **Reward/eval allocator mismatch** | 08-24 used ``use_hard_band=false`` for the reward allocator (long-short, gross=1) but ``true`` in validation (de-levers to cash).  The reward allocator's vv improvement (0.73 → 0.47) does not transfer to validation (val var_deployment vv stayed ~0.55–0.59).  Use the same allocator config in both when interpreting vv. |
| **One-sided hinge leaves flattening unguarded after a peak refresh** | 08-28 run (``logp_target=-100``, one-sided hinge, ``kl_beta=0.2``): the hinge successfully bounded logp (−50..−82, no concentration runaway) while advantage stayed ~0.  But the refresh at step 657 (val sharpe 0.038) ratcheted the baseline to a *noise peak*; the model regressed, ``advantage`` went persistently negative (−0.04), REINFORCE flattened logp (−57 → **−198**), the hinge went **inert** below −100 (``hinge`` → 0), and recall@10_q95 collapsed 8.3 % → 0.8 % (val sharpe −0.038).  Root cause: refreshing to a single peak makes the baseline unbeatable → structurally negative advantage.  Fix: symmetric hinge (``relu(logp − target) + relu(target − logp)``) so flattening below −100 is also penalised; longer-term, avoid ratcheting the reference to a single noisy peak (sustained-improvement gate or soft/EMA reference). |
| **RL score function scored the wrong series identity (fixed 09-06)** | Before the fix, each step drew a random series subset **after** sampling: ``model.sample()`` ran the full 1,302-joint (O(1302²) cost + attention noise), the reward used the correctly-sliced subset columns, but the REINFORCE score function called ``model.loss()`` on those shuffled columns — which looked up embeddings ``0..len(subset)-1`` ("identity by column order"), i.e. the wrong assets, under a different (subset-conditional) joint than the one that produced the samples.  Advantage and logp were thus estimated off-policy and mis-labelled.  Fix: ``subset_scope`` + ``TACTiSModel.sample_subset`` (see below) slice history **before** sampling and rebind the per-series embedding lookups to the actual subset, so sampling, reward and gradient share one subset-conditioned joint.  Attention drops to O(100²) per step (``rl.series_subset``); per-step wall time ~15s → ~6s.  **logp level shifts up** (see §Subset-conditioned sampling): the initial checkpoint's on-policy ``rl/logp`` is ≈ **+25.4** (the legacy −100 was the deflated wrong-identity scale), so ``logp_target`` must be re-centered to ≈ +25. |
| **Linear hinge loses to REINFORCE by ~9×/nat — logp collapses even post-fix (fixed → L2 hinge, 09-06)** | 09-06 run with ``logp_target=25``, symmetric *linear* hinge, ``logp_penalty_weight=0.01``: first 300 steps ``rl/logp`` p10/med/p90 = +12.7 / +28.6 / +42.7 (healthy), but by step 2190 the median was **−225** (final-300 p10 −326) and **23 % of all steps had ``logp < −100``**, with mean advantage −0.09 on those steps vs −0.004 elsewhere — i.e. negative advantage driven flattening that the constraints couldn't hold.  Per-nat of logP, REINFORCE pushes ``|adv| ≈ 0.09`` while the linear hinge pushes only ``weight = 0.01``; the L2 hinge's ``2wΔ`` gradient reaches the same strength at Δ≈25 and dominates at collapse scales (Δ≈250 → term ≈ 60 vs REINFORCE ≈ 10–30).  Term-scale audit of that run: REINFORCE median ≈ −0.22 (p10 −30.7), anchor ``0.2·sq`` ≈ 0.36, linear hinge ≈ 0.20 → constraints were ~10 % of REINFORCE on typical steps but orders short in the left tail.  Remedy: Euclidean hinge (``(logp_sum − target)²``, ``logp_penalty_weight=0.001``).  Note ``rl.clip_grad`` is a **global L2 clip on the trainable grads only** (rescales the whole vector if ‖g‖>1) — it caps per-step magnitude but preserves direction, so it cannot stop a persistent negative-advantage drift by itself. |

### RL fine-tuning — `post_training.py`

Post-train RL fine-tuning to align TACTiS embeddings with portfolio reward.
Freezes the transformer/decoder, trains only ``flow_series_encoder`` and
``copula_series_encoder`` with sharpe or mean_return as reward.

Uses **policy gradient** (REINFORCE-style): the model's embeddings serve as the policy
parameters; ``model.sample()`` generates portfolio outcomes (reward from the reward
allocator, default ``var_deployment``); ``model.loss()`` provides the
differentiable log-probability.
The loss is ``−A · log P(pred|hist) + β · ½ (log π_θ − log π_ref)² + γ · (log π_θ − logP_target)²``,
where the advantage ``A = reward − reward_ref`` subtracts the reward of a **frozen reference
model** (the pre-RL checkpoint) under the same allocator, the squared log-density constraint
is evaluated on samples drawn from **both** the current and reference models (a symmetric
divergence), and the **Euclidean (L2) hinge** penalises the current model's log-probability deviating
from the ``logp_target`` sweet spot in **both** directions — over-concentration *and*
over-flattening — so the bound holds regardless of the sign of the advantage.  No actor-critic
or value network — the reference model is the baseline.

Why a reference-model baseline: the equal-weight portfolio baseline was systematically
positive (market drift 2008–2024 + diversification), so ``reward − baseline`` was negative
~62 % of steps and REINFORCE kept pushing log-probability down, flattening the predicted
distribution.  The reference model's own reward is a beatable, model-relative baseline.
The symmetric anchor (current + reference samples) is required because a reference
log-density evaluated only at the *current* model's samples drifts down in lockstep with
the model and never resists flattening.

```bash
python post_training.py rl.resume_ckpt=outputs/.../custom_tactis_medium.pth
python post_training.py rl.resume_ckpt=<path> rl.epochs=100 rl.reward=sharpe
```

Key design choices:

| Choice | Why |
|---|---|
| Policy gradient (REINFORCE) | Simple, no critic network — reward directly backpropagated |
| Train embeddings only | Smallest parameter set that controls per-series signal |
| Freeze transformer/decoder | Preserves learned cross-series attention patterns |
| Baseline | Frozen reference model's own reward under the same allocator (``advantage = reward − reward_ref``) — a beatable, model-relative baseline instead of equal-weight |
| Anchor loss | Symmetric squared log-density constraint to the frozen reference, evaluated on samples from **both** current and reference models — resists distribution flattening |
| Logp hinge | **Euclidean (L2)** ``(log π_θ − logp_target)²`` — penalises over-concentration *and* over-flattening around the sweet spot (``logp_target ≈ +25`` on the post-fix on-policy scale; the legacy −100 was the deflated wrong-identity scale).  At the current ``logp_penalty_weight=0.001``: typical-step term ≈ 0.13 vs REINFORCE ≈ 0.7 (secondary), but a collapse (Δ≈250) → term ≈ 60 — the L2 gradient ``2wΔ`` overtakes the REINFORCE force ``≈ |adv|`` exactly where it matters (linear ‖Δ‖ lost by ~9×/nat: the 09-06 linear-0.01 run collapsed logp +28 → −225 over 2190 steps).  One-sided was insufficient for the opposite reason — a negative-advantage phase (after a refresh) flattened logp unchecked |
| Series subset per step | Random 100-series sample per step, subset-conditioned end-to-end (see §Subset-conditioned sampling below) — the encoder, copula and marginal all run on only the subset, so attention is O(100²) rather than O(1302²) |
| Reward allocator | ``var_deployment`` with ``use_hard_band=false`` (long-short MV + beta/CVaR penalties, gross=1); ``var_optimizer`` is the long-only alternative |
| Validation | Full ``compute_validation_metrics`` with all allocators once per epoch (plus once on the initial checkpoint); reference baseline refreshed when val sharpe beats the margin |

If stronger RL is needed later, natural extensions:
- **Learned value baseline / actor-critic**: a small MLP critic ``V(s; emb)`` with
  advantage ``R − V(s)`` could reduce variance beyond the frozen-reference baseline
- **PPO**: clip embedding updates to prevent destructive large steps
- **Entropy bonus**: add ``−H(w)`` to encourage diversified allocations

Config under ``rl:`` (top-level in ``cfg/config.yaml``):

```yaml
rl:
  resume_ckpt: null              # Required: path to stage-2 .pth
  epochs: 2
  steps_per_epoch: null          # cap gradient steps per epoch (null = full pass)
  batch_size: 16
  micro_batch: 16
  lr: 1e-4
  clip_grad: 1.0
  series_subset: 100
  n_samples: 10                  # MC samples per window (25 gives tail_size=2 for custom_reward_*)
  n_chunks: 1                    # gradient-accumulation chunks for the score function
  reward: sharpe                 # sharpe | mean_return | combined | custom_reward_product
  reward_alpha: 0.5
  reward_scale: 1.0              # multiplier for custom_reward_product: (1 - vv) * sharpe * scale
  kl_beta: 0.1                   # squared log-density constraint weight (anchor to the reference model)
  logp_target: -100.0            # sweet-spot log-probability (rl/logp units); RE-CENTER to ≈ +25 on the post-fix on-policy scale (see §Subset-conditioned sampling)
  logp_penalty_weight: 0.001     # L2 hinge weight: loss = w·mean((logp_sum−logp_target)²); typical Δ≈20 → 0.13, collapse Δ≈250 → ~60
  baseline_refresh_margin: 0.01  # promote current model to reference baseline if val sharpe exceeds this margin
  freeze_embeddings: true        # flow_series_encoder + copula_series_encoder
  freeze_encoder: true           # input_encoder, time_encoding, transformer
  freeze_decoder: false          # decoder.marginal + decoder.copula
  allocator:                     # same format as eval.allocators entries
    - name: var_deployment       # use_hard_band=false → long-short, gross=1
      risk_aversion: 1.0
      soft_cvar_weight: 1.0
      n_steps: 100
      optimizer_lr: 0.01
      beta_penalty_weight: 0.1
      weight_lower: -0.05
      weight_upper: 0.05
      max_assets: 50
      return_lower: -0.01
      return_upper: 1.0
      band_confidence: 0.95
      turnover_coef: 0.0
      use_hard_band: false
    # var_optimizer (long-only, max mean s.t. CVaR soft constraint) is the alternative:
    #   n_steps: 100, optimizer_lr: 0.05, soft_cvar_weight: 1.0, max_loss: 0.01,
    #   constraints: {max_assets: 50, max_weight: 0.05}
  log_every_steps: 1             # wandb per-step logging frequency
  topk_checkpoints: 4
```

**Reward options** (``rl.reward``): each is applied per timestep and averaged over
``pred_len``; the frozen reference model's reward (the advantage baseline) uses the same
formula on its own sampled portfolio.

| Reward | Formula |
|---|---|
| ``sharpe`` | ``mean(ret) / (std(ret) + 1e-8)`` |
| ``mean_return`` | ``mean(ret)`` |
| ``combined`` | ``reward_alpha · sharpe + (1 − reward_alpha) · mean_return`` |
| ``custom_reward_product`` | ``reward_scale · sharpe · (1 − vv)^is_pos``, ``is_pos = 1`` if ``sharpe > 0`` else ``0``; ``vv = mean(ret < var)`` |

``var`` is the ex-ante VaR of the **reward allocator's** own portfolio (``rl.allocator``,
default ``var_deployment``) via ``BaseAllocator.compute_var_cvar`` — *not* the
validation allocators (``proportional``/``var_optimizer``/``var_deployment``) whose
``val/portfolio/*/var_violation`` appears in the logs.  Per-step ``rl/var_violation`` is
logged to wandb.

**Product form gating**: the ``(1 − vv)`` factor only applies when ``sharpe > 0``
(``is_pos = 1``).  On non-positive sharpe the exponent is ``0``, so the factor drops out
and the reward is just ``reward_scale · sharpe`` — this removes the sign-inversion
where high violation could *increase* the reward on negative-sharpe steps.  Also note VaR
is degenerate at low ``n_samples``: with ``n_samples=10, var_alpha=0.05`` the tail size is
1 (single worst sample), so ``vv`` is noisy — raise ``rl.n_samples`` (e.g. 25) for a more
stable signal.

The reward allocator can be toggled via CLI without code changes:

```bash
python post_training.py rl.allocator[0].name=proportional
python post_training.py rl.allocator[0].name=mv_neutral_cvar
python post_training.py rl.allocator[0].name=var_optimizer rl.allocator[0].soft_cvar_weight=2.0
python post_training.py rl.allocator[0].name=var_deployment rl.allocator[0].use_hard_band=true
```

**Freeze toggles** control which TACTiS components receive gradients:

| Toggle | Affected params | Default |
|---|---|---|
| ``freeze_embeddings`` | ``flow_series_encoder``, ``copula_series_encoder`` | ``true`` |
| ``freeze_encoder`` | ``{flow,copula}_input_encoder``, ``{flow,copula}_time_encoding``, ``{flow,copula}_encoder`` | ``true`` |
| ``freeze_decoder`` | ``decoder.marginal``, ``decoder.copula`` | ``false`` |

At defaults (embeddings + encoder frozen, decoder trainable): 32 trainable params
in ``decoder.marginal.*`` + ``decoder.copula.*`` — the smallest possible set, preserving
all learned representations.

Training is step-based, not epoch-based.  Each ``global_step`` is one ``model.sample()``
call (micro-batch).  The DataLoader is recreated each epoch for a fresh shuffle.
Validation runs once per epoch (and once on the initial checkpoint before training): the
initial validation seeds ``ref_val_sharpe``, and subsequent epochs promote the current
model to the frozen reference baseline when
``val_sharpe > ref_val_sharpe + baseline_refresh_margin``.

| Setting | Steps/epoch | Time/epoch |
|---|---|---|
| ``epochs=2`` (full passes) | ~440 | ~110 min |
| ``epochs=1, steps_per_epoch=200`` | 200 | ~50 min |

Checkpoints saved to ``outputs/<timestamp>/checkpoints/`` with step-based
filenames (``rl_step{step:06d}_sharpe{val:.4f}.pt``).  Top-4 kept by val ``var_deployment`` sharpe.

**Memory note**: each step samples and scores **only the ``rl.series_subset`` series**
(100) — ``model.sample_subset()`` slices history to the subset and
``subset_scope`` (``model.py``) redirects the per-series embedding lookups to the
*actual* subset rows, so both the reward samples and the REINFORCE score function
come from the same subset-conditioned joint (attention O(100²), marginal inverse
O(100)).  Validation/eval still sample the full 1,302-joint via
``compute_validation_metrics``.  Each step still draws ``ref_model.sample_subset()``
(frozen reference) and evaluates ``model.loss()`` on both the current and reference
sample sets, so the score-function graph is roughly doubled vs. a single-sample step.
At ``batch_size=16`` with gradient tracking, per-step memory is ~4 GB on GPU — tune
``n_chunks`` (e.g. 2) to reduce peak memory at the cost of speed.

**Step monitoring**: tqdm bar shows reward, loss, log-probability (``logp``), the
anchor distance (``sq``), the variance hinge (``hinge``), and gradient norm per step.
Wandb logs the corresponding ``rl/*`` metrics (incl. ``rl/logp_hinge``) at
``log_every_steps`` frequency (set to 1 for continuous time-series).

### Subset-conditioned sampling (§why scale is unchanged)

`post_training.py` samples and scores a fresh random ``rl.series_subset`` (100) series per
step, but its **identity** is fixed by ``subset_scope``/``sample_subset`` in ``model.py``:

- Upstream ``TACTiS.sample()``/``loss()`` look up per-series embeddings as
  ``series_encoder(torch.arange(num_series))`` — they infer series identity purely from
  **column order**.  Slicing history to a subset and calling the sliced columns directly
  would score them under the embeddings of series **0..len(subset)-1**, not the sampled
  subset.
- ``subset_scope(models, subset_idx)`` temporarily swaps each model's two
  ``series_encoder`` modules for a ``_GatherSeriesEmbedding`` wrapper that ignores the
  ``arange`` argument and returns ``base(subset_idx)`` rows; it restores the originals on
  exit (also on exception).  It accepts one model or several (current + frozen reference).
- ``TACTiSModel.sample_subset(...)`` slices ``hist_value`` to the subset, enters the scope
  and runs the vanilla ``sample`` — the encoder, copula and marginal all operate on only
  the subset, so attention is O(100²) and the marginal inverse is O(100).
- Both the **reward samples** and the **score-function log-density** now come from the
  same subset-conditioned joint (``post_training.py`` wraps the whole step in one scope).

This fixes two earlier defects: (1) the old code sampled the full 1,302-joint then
sliced the reward columns after the fact, wasting O(1302²) attention per step; (2) the
score function evaluated ``model.loss()`` on the sliced columns under sequential
``0..len(subset)-1`` embeddings, i.e. the **wrong per-series identity** — a REINFORCE
gradient pointing at a distribution that never generated the actions.  ``tests/test_subset_sampling.py``
regresses the identity property: gradients on ``flow_series_encoder.weight`` /
``copula_series_encoder.weight`` must be exactly zero outside the subset.

**logp level shifts up — re-center ``logp_target``**: the series dimension is a *mean*
(``loss_normalization: "series"`` divides by ``ns = len(subset)``), so the subset-size
divisor is the same before/after and the per-step estimator shape is unchanged
(measured identical at ``series_subset`` 90 and 100).  But the absolute level moves a
lot: the old score was systematically *deflated* by wrong-identity + off-policy
evaluation, while the corrected score is the model's own on-policy density.  Measured
on the initial pre-RL checkpoint (08-15 ``top_stage2_epoch0125``, 40 fresh 100-series
subsets, ``n_samples=10``, train-mode dropout): ``rl/logp`` mean **+25.4**, std 9.2,
p5–p95 ≈ 11–39.  ``logp_target`` should therefore be re-centered to ≈ **+25** (the
legacy −100 was calibrated on the deflated scale and the L2 hinge would
actively fight REINFORCE).  ``rl/logp`` scales ≈ linearly with ``n_samples`` (target
≈ 63 at ``n_samples=25``); it is insensitive to the subset size.  A wrong-identity /
random-init probe reads ≈ −98.5 and is *not* comparable to the trained-checkpoint
scale.

Checkpoints saved to `outputs/<date>/<time>/<microseconds>/checkpoints/` (top-k + last per stage, isolated per run via `hydra.job.chdir: true`).  Final model saved to `outputs/<date>/<time>/<microseconds>/{dataset}_{model}.pth`.

- `last_stage{stage}.pt` — most recent epoch (for resume)
- `top_stage{stage}_epoch{e:04d}_varviol{:.4f}_sharpe{:.4f}.pt` — top-k best-val checkpoints (configurable via `training.topk_checkpoints`, default 10); the suffix encodes `var_deployment` var_violation and sharpe for checkpoint selection
- `outputs/<date>/<time>/<microseconds>/{dataset}_{model}.pth` — final model artifact (best stage-2 epoch, with real best_val_loss)

`model.py`: TACTiSModel wrapper with NaN-aware loss, delegates `sample`, `set_stage`, `initialize_stage2` to the underlying ``TACTiS`` instance.  ``build_model(cfg, num_series)`` validates model config keys against ``TACTiS.__init__`` signature, then creates and returns a ``TACTiSModel`` on the configured device.

Note: ``initialize_stage2()`` creates new modules (copula encoder, embedding, input encoder) on CPU.  ``switch_to_stage2`` in ``trainer.py`` calls ``model.to(device)`` after ``initialize_stage2()`` to move them to GPU.

**Train/test strategy — two phases:**

| Phase | Train/val split | Test set | Purpose |
|---|---|---|---|
| Development | Walk-forward chronological (80/10/10) | Last 10% of historical data | Observe forecasting decay in a nowcasting scenario |
| Production | Shuffled split on full historical data (2008–2024) | Unseen 2025–current data | Final model for forward deployment |
| Mitsui production | Train/val from `train.csv` (walk-forward) | Official `test.csv` (134 days) | Verify against competition test set |

Currently in development phase — walk-forward splits let us measure how quickly forecast accuracy degrades across time.

For mitsui production evaluation, set `use_official_test: true` in the dataset config:
```bash
python train.py dataset=mitsui dataset.use_official_test=true
```

## Evaluation

Post-training test evaluation via `eval.py`.  Configurable in `cfg/config.yaml` under the
``eval:`` section.

### Test evaluation

```bash
python eval.py eval.checkpoint=outputs/2026-08-05/12-08-01-708369/custom_tactis_medium.pth
python eval.py eval.checkpoint=<path> eval.n_samples=50 eval.sample_micro_batch=2
```

Reports per-timestep recall@k (mean and q95) at multiple k values plus portfolio metrics
per allocator, including var_optimizer diagnostics (nonzero weights, cash allocation,
effective N, VaR violation rate).  Results saved as ``test_metrics.json`` alongside the
checkpoint (or ``eval.output_dir``).

Key config options (``eval.``):
- ``checkpoint: null`` — path to ``.pth`` file (required)
- ``n_samples: 10`` — MC samples per window
- ``recall_k: [1, 5, 10, 20, 50]``
- ``sample_micro_batch: 8``, ``batch_size: 64``
- ``allocators`` — same format as ``training.metrics.allocators``
- ``output_dir: null`` — defaults to checkpoint parent directory

### Model loading for evaluation

When the checkpoint was saved after stage 2 (copula active), ``eval.py`` must call
``model.initialize_stage2()`` before ``load_state_dict`` because ``build_model``
uses ``cfg/model/tactis_medium.yaml`` with ``skip_copula: true`` (stage 1 model).
The copula modules are created by ``initialize_stage2()``, then weights load with
matching keys.

### Sampling performance

On GPU with 1,302 series and copula active, per-call timing at ``batch_size=4``:

| n_samples | Time | GPU memory |
| 2 | 14s | 0.84 GB |
| 10 | 14s | 0.88 GB |
| 20 | 20s | 0.92 GB |
| 50 | 23s | 1.07 GB |

The main cost is the DSF marginal inverse CDF binary search (~100 iterations per
series per sample).  Copula attention ``O(S²)`` is shared across samples, but at
higher ``n_samples`` the binary search dominates.  Time scales roughly linearly
above 10 samples.  For eval at ``n_samples=10`` the full test set (418 windows ×
batch=64, micro_batch=8) completes in ~20 minutes.  For RL training, keep
``n_samples`` low (10) — the reward signal from proportional allocation is
sufficient and the gradient comes from ``model.loss()``.

### Allocator diagnostics

``var_optimizer`` and ``var_deployment`` report additional metrics not in
``compute_validation_metrics``:
- ``nonzero_weights`` — average number of positions with |w| > 1e-8
- ``cash_allocation`` — average portfolio weight allocated to cash
- ``effective_n`` — 1 / Σw_i² (inverse Herfindahl), higher = more diversified
- ``var_violation`` — fraction of windows where realized return < ex-ante VaR
- (var_deployment only) ``gross_exposure``, ``net_exposure`` — total |long|+|short| and net long−short book

## Dataset classes

Two `torch.utils.data.Dataset` implementations in `data/dataset.py`:

```python
from data.dataset import MitsuiReturnDataset, CustomReturnDataset
```

### MitsuiReturnDataset (mitsui)

Multivariate commodity/futures/FX returns. **Recommend ``price_column="close"``** — uses all
143 assets (LME metals, JPX futures, FX pairs, US ETFs) → 143 TACTiS series.  When
``price_column="open"`` only JPX futures are used (~40-60 series); LME, FX and US ETF assets have
no open prices and are silently dropped (a warning is printed).  1,960 return timesteps
(1,961 trading days).

*Note from Human Author:* this is not ideal and should be fixed. Currently, we are focusing on the model training the custom dataset and we'll leave it as it and leave investigating the attribute name for Mitsui for fixing the problem for future contributions. 

```python
ds = MitsuiReturnDataset(hist_len=21, pred_len=4, split="train", price_column="close")
# ds.num_series → 143
# ds[0] → (Tensor[143, 21], Tensor[143, 4], Tensor[143, 4])  # hist, pred, pred_mask
```

### CustomReturnDataset (custom stock data)

US + CN stocks from yfinance/baostock. Prefers the merged `all_stocks.csv`, falls back to per-ticker CSVs. 1,302 tickers → 1,302 TACTiS series. 4,416 return timesteps (2008–2024).

```python
ds = CustomReturnDataset(hist_len=21, pred_len=4, split="train", price_column="close")
# ds.num_series → 1,302
# ds.timesteps → 4,416 (2008-2024)
```

Both datasets:
- Single price column mode: `price_column="close"` or `price_column="open"` (default: `close`)
- Compute arithmetic returns: `returns[t] = (price[t] - price[t-1]) / price[t-1]`
- NaN → 0.0 filled; per-series validity mask stored in `self.mask` (separate from `self.data`)
- ``np.isfinite`` catches both NaN and inf (division-by-zero → inf); returns are zero-filled for all non-finite values
- Per-series z-score normalization (stats from train split only, computed via mask)
- Walk-forward splits via index bounds (80%/10%/10%)
- `__getitem__` returns 3-tuple `(hist, pred, pred_mask)` — deterministic sliding windows; DataLoader with `shuffle=True` randomizes order
- `pred_mask` is boolean — ``True`` = real prediction target, ``False`` = NaN-filled (excluded from loss)

### Token values ($x_{ij}$)

Each TACTiS token is an arithmetic daily return from the configured price column:
```
returns[t] = (price[t] - price[t-1]) / price[t-1]
```

### `model.py` — TACTiSModel wrapper

```python
from model import TACTiSModel, build_model
```

Wraps the upstream `TACTiS` with NaN-aware loss masking.  Key additions:

- `model.loss(hist_time, hist_value, pred_time, pred_value, nan_pred_mask=None)` — replicates the TACTiS encoding pipeline but overlays a NaN exclusion mask before calling `decoder.loss()`.  Positions where `nan_pred_mask` is ``False`` are treated as observed (not predicted), so they contribute zero marginal and copula loss.
- `model.sample(...)`, `model.set_stage(...)`, `model.initialize_stage2()` — delegate to the underlying TACTiS instance.
- `model.state_dict()`, `model.load_state_dict()`, `model.named_parameters()`, etc. — delegate transparently.
- `model.to(*args, **kwargs)` — forwards all arguments to the underlying TACTiS (supports mixed-precision, non-blocking transfers).

NaN masking is applied only during **stage 1** (marginal training).  Stage 2 (copula training) requires uniform prediction windows, so `nan_pred_mask` is ``None`` during stage 2.

## Transfer learning between datasets

TACTiS uses `nn.Embedding(num_series, dim)` for series identity — one learned vector per asset index. This makes the model dataset-specific at the embedding level, but the transformer + decoder weights are series-count-agnostic.

The TACTiS-1 paper prescribes a cold-start strategy: train on one dataset, swap the embedding layer (a covariate lookup), fine-tune on another.

```
Phase 1: Train on mitsui (143 series)
    → saves outputs/<date>/<time>/mitsui_tactis.pth

Phase 2: Transfer to custom stocks (1,302 series)
    1. Create model_B: new nn.Embedding(1302, dim) + same transformer/decoder dims
    2. Copy transformer & decoder weights from Phase 1 checkpoint
    3. Freeze all except flow_series_encoder + copula_series_encoder
    4. Warmup: train embeddings only for N epochs
    5. Unfreeze, fine-tune full model at lower LR
```

Ideas from human author:

- Alternatively, we can avoid the covariate lookup entirely: ask an LLM to describe each asset, obtain a sentence embedding (e.g. via mini-LM) as the covariate for each series. This solves the cold-start problem better since new assets with sparse data still get a meaningful embedding from their description. The covariate dimension is typically small (5–64), so we should experiment with an MLP to map the sentence embedding down vs. feeding the raw embedding directly. For the imminent run, stick with standard transfer learning.

- Or for zero-shot performance, train a covariate representation(init from the mean of features) that will averagely for every assets. (Can do this in RL part during training for production) This can be useful for predicting return of new stocks.

- For dataset with certain metadata features(say, another kaggle competition), train a MLP that can map metadata to covariate dim.

Transferability can be stress-tested on recently-listed stocks absent from the training window: SATS (EchoStar, ex-DISH), FDXF (FedEx Freight, IPO May 2026), and WBA (Walgreens, re-listed post-buyout). These three have no data before 2024-2025 and are excluded from the current 2008-2024 download — they serve as held-out transfer targets.

### TACTiS layer map

| Layer name patterns | Component | Transfer action |
|---|---|---|
| `flow_series_encoder` | Per-series embedding | **Replace** |
| `copula_series_encoder` | Per-series embedding | **Replace** |
| `flow_time_encoding` | Temporal position encoding | Copy |
| `flow_input_encoder` | Per-token MLP | Copy |
| `flow_encoder` | Transformer layers | Copy |
| `decoder.marginal.*` | DSF marginal decoder | Copy |
| `copula_time_encoding` | Temporal position encoding | Copy |
| `copula_input_encoder` | Per-token MLP | Copy |
| `copula_encoder` | Transformer layers | Copy |
| `decoder.copula.*` | Attentional copula decoder | Copy |

## Run tests

There are no unit/integration tests in this repo. When adding tests, prefer `pytest` with the standard `pip install pytest` and run via:

```bash
pytest tests/
pytest tests/test_something.py::test_case -v
```

## Directory layout

```
data/
  dataset.py       # MitsuiReturnDataset & CustomReturnDataset
  custom-data/     # data pipeline: download, merge, config
    config.py      # shared paths, constants, utility functions
    download_us.py # yfinance US stock downloader
    download_cn.py # baostock China stock downloader
    merge.py       # merge individual CSVs → unified dataset
    fetch_components.py  # fetch index component ticker lists
    components/    # ticker CSVs (hs300, sp500, nasdaq100, etc.)
    raw/           # downloaded OHLCV CSVs per ticker + all_stocks.csv
  mitsui-commodity-prediction-challenge/  # multivariate commodity/spread forecasting
    train.csv              # 1,961 days, 557 features (LME metals, JPX futures, US ETFs, FX OHLCV)
    train_labels.csv       # 424 targets (asset-pair return spreads, lags 1-4, ~10% nulls)
    target_pairs.csv       # maps target_N → pair formula + lag
model.py           # TACTiSModel wrapper (NaN-aware loss, delegates)
train.py           # Hydra training entry point (+ seed application)
trainer.py         # 2-stage training loop, loss, validation, checkpoint I/O
eval.py            # Test-set evaluation (per-timestep recall, portfolio diagnostics)
post_training.py   # RL fine-tuning (freeze transformer, train embeddings via portfolio reward)
metrics.py         # Validation metrics (recall@k, portfolio)
portfolio/         # Allocator package (proportional, var_aware, var_optimizer, mv_neutral_cvar, mv_long, var_deployment)
  constraints.py   # PortfolioConstraints dataclass
  base.py          # BaseAllocator ABC + constraint projection
  proportional.py  # ProportionalAllocator
  var_aware.py     # VaRAwareAllocator
  var_optimizer.py # VaROptimizer (MC gradient descent)
  mv_neutral.py    # MeanVarianceNeutralCVaRAllocator (dollar-neutral, MV + CVaR)
  mv_long.py       # MeanVarianceLongAllocator (long-only, MV + CVaR)
  var_deployment.py # VarDeploymentAllocator (long-short MV + beta/CVaR + return band, cash dial)
backtest.py        # Venue 0 runner (rolling-partition strategy + BacktestExecutor, mirrors eval.py)
trading/           # Shared application layer (Venue 0 + live Venues 1/2 executors)
  base.py          # ExecutorBase ABC, PositionManager, Order/FillReport
  data.py          # BacktestData (date-indexed data, frozen universe + norm stats)
  strategy.py      # RollingPartitionStrategy (span-forecast rolling partitions)
  portfolio.py     # Partition (per-partition cash/NAV book + snapshots/roll records)
  ledger.py        # CsvLedger (append-only durable CSV history, shared by all venues)
  costs.py         # CostModel / MarketCost / build_cost_model (shared by backtest + IBKR)
  live.py          # LiveGuard (dry-run default, two-factor live gate, kill-switch, notional cap)
  samplers.py      # ModelSpanSampler / RandomSpanSampler
  venues/
    backtest.py    # BacktestExecutor (shares/cash book, open(t+1) fills, costs)
    ibkr.py        # IBKRExecutor + IBKRBroker (Venue 1, US + CN Stock Connect, ib_async)
    xueqiu.py      # XueqiuExecutor + XueqiuClient (Venue 2, CN long-only, easytrader, full rebalance)
cfg/               # Hydra configs
  config.yaml      # root defaults (dataset, model, training, wandb, hydra)
  application.yaml # live venues: live:/ibkr:/xueqiu: (separate from training/backtest)
  dataset/         # custom.yaml, mitsui.yaml
  model/           # tactis_medium.yaml
requirements-app.txt # application-only deps (ib_async, easytrader) — not needed for training/backtest
outputs/           # Hydra run dirs — one per experiment (gitignored)
  YYYY-MM-DD/
    HH-MM-SS-mmmmmm/
      checkpoints/   # top-k + last per stage
      {dataset}_{model}.pth  # final model artifact
      train.log       # console output
      .hydra/         # config snapshot
temp_scripts/      # one-off analysis scripts (gitignored)
```

## Import conventions

```python
# 1. Standard library
import argparse
import time
from pathlib import Path

# 2. Third-party (alphabetical, common aliases)
import numpy as np
import pandas as pd
import torch
import yfinance as yf

# 3. Local modules (bare name — scripts run from their directory)
import config
```

- Use `from pathlib import Path`; root paths via `Path(__file__).resolve().parent`.
- Never mix `import` and `from ... import` in the same import group.

## Code style

### Formatting

- 4-space indentation, no tabs.
- Line length: 100 characters preferred, 120 max (no formatter configured).
- Trailing commas in multi-line dicts/lists/calls.
- Two blank lines between top-level functions/classes, one blank line between methods.
- UTF-8 encoding, unix-style LF endings.

### Naming conventions

| Thing | Convention | Example |
|-------|-----------|---------|
| Files / modules | `snake_case` | `download_us.py` |
| Functions / methods | `snake_case` | `fetch_ohlcv()`, `sample_batch()` |
| Variables | `snake_case` | `hist_len`, `out_dir` |
| Constants (module-level) | `UPPER_CASE` | `START_DATE`, `MAX_RETRIES` |
| Classes | `PascalCase` | `TACTiS` |
| Private helpers | `_leading_underscore` | — |

### Type annotations

Use type hints on function signatures. Prefer modern syntax (`list[str]`, `dict[str, float]`) over `typing.List`/`typing.Dict`. Import from `typing` only when needed (`Optional`, `Generator`, etc.). Annotate module-level constants (e.g. `US_TICKERS: list[str] = []`).

### Function structure

Scripts use the `main()` + `if __name__ == "__main__"` pattern:

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--output", default=None, help="...")
    args = parser.parse_args()
    # logic ...

if __name__ == "__main__":
    main()
```

Keep functions small and single-purpose. Use keyword arguments in calls with 3+ arguments for readability.

### Docstrings

Triple-quoted docstrings for modules (first lines of file) and public functions. One-line summary first, details after a blank line. NumPy/Google-style parameter docs are fine but not enforced.

### Error handling

- Catch specific exceptions (`except Exception` only at the outermost retry/download boundary).
- Download scripts use `config.MAX_RETRIES` with `time.sleep(5)` between retries.
- Exit with `sys.exit(1)` on unrecoverable errors after printing to stderr (or `print()` — no logger configured).
- Use `pathlib.Path.mkdir(parents=True, exist_ok=True)` for output directories.

### Configuration

Shared settings live in `data/custom-data/config.py` as module-level constants. Scripts import this module by bare name (`import config`) since they run from the `custom-data/` directory. Ticker lists can also be loaded from CSV at runtime via `config.load_tickers_from_csv()`.

### PyTorch

- Always handle device placement: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Use `.float()` when converting numpy → torch tensors for model input.
- Call `model.eval()` before sampling/inference.
- Explicit `.to(device)` after model instantiation and parameter groups.

### Dependencies

Add new dependencies to `requirements.txt` (unpinned is OK for now). The `tactis` package is installed separately from PyPI (`pip install tactis`), not from a local source tree. The commented-out git reference in requirements.txt references `github.com/Servicenow/tactis.git@tactis-2`.

## Application

This repo's trading model is wired to LIVE venues through a shared
**portfolio-target contract**: the model/allocator emits a target-weight
vector `dict[str, float]`; a shared **PositionManager** layer reads current
positions, computes the diff, applies risk filters (suspended / limit-D /
delisted / shortability), then dispatches to a broker executor behind ONE
interface. Each venue is just one executor:

- **Venue 0 — Backtest (2024–2026)** — in-repo simulation of the full stack on
  recent out-of-sample data; first consumer of the shared `trading/` layer.
  See below.
- **IBKR (Duke FinTech)** — full capability: long + short (real short-selling),
  global markets, options/futures. Main multi-venue competition entry.
  **Executor implemented** (`trading/venues/ibkr.py`): US (long+short) + China
  A-shares via Stock Connect (long-only, 100-share lots); dry-run default,
  daily EOD signal → at-the-open orders, IBKR cost schedule reused for pre-trade checks.
- **Xueqiu portfolio (雪球组合)** — long-only, one portfolio per market
  (cn/us/hk), <=30 holdings, 1% min weight, cookie auth, full-rebalance API.
  **Executor implemented** (`trading/venues/xueqiu.py`, CN): dry-run default,
  target-based full rebalance via `easytrader`.

### Venue 0 — Backtest (2024–2026)

Purpose: exercise strategy → PositionManager → executor end-to-end without broker
complexity, and measure raw signal on 2025-01 → 2026-09 out-of-sample data before the
Duke FinTech window. Data extension, strategy, and accounting live here; IBKR/Xueqiu
reuse the same `trading/` layer.

#### Status (2026-09-18)

- **Base layer implemented and tested**: `trading/` (base, data, strategy, samplers, ledger,
  portfolio, venues/backtest) + top-level `backtest.py` runner + `backtest:` config section.
  27 CPU pytest invariants pass (`pytest tests/`); mock-model smoke on the real 1,302-series
  data confirms turnover ≈ 1/N, the NAV identity, bounded gross, and per-partition CSV history.
- **Data extended and trimmed**: `all_stocks.csv` now spans 2008-01-02 → **2026-08-10**
  (4,826,631 rows, 1,302 tickers) — the 2025–26 rows came from `data/custom-data-test/`
  (US to 2026-08-07, CN to 2026-08-10); the degenerate 2026-08-10 → 09-16 tail (a single
  CN ticker `sh.600000`) was trimmed out. `base_end_date=2024-12-31` keeps universe + norm
  stats frozen.
- **Correctness fixes** (2026-09-18):
  1. **Reconciliation** (`trading/strategy.py`): partition books are re-anchored to the
     executor's actual position dollars + cash every day, so skipped fills / open-close gaps /
     reset re-attribution can no longer make position tallies drift (that bug had silently
     levered the account to ~9× gross in the first run).
  2. **Allocator-level CN short ban** (`portfolio/var_deployment.py`): `long_only_mask`
     (from `BacktestData.market_mask("cn")`) clamps CN names' per-name lower bound to 0 —
     the optimizer can no longer plan phantom CN shorts; its CVaR/band/beta model matches the
     executed book. `BaseAllocator.long_only_mask` defaults to `None`, so eval/val/RL are
     unchanged.  User's PositionManager clamp stays as defense-in-depth.
  3. **Turnover penalty = `τ′·c`**: `turnover_coef` × `max(commission_rate, 3e-4)` replaces
     the fixed `turnover_weight` — the allocator anticipates commissions.
  4. **`min_valid_series` guard** (default 25): sparse windows (early history / symbol tail)
     stay in cash instead of concentrating into a handful of names.
  5. **Stable reconcile attribution**: per-name book distribution uses each partition's
     **gross** share of the actual position, not the signed `book_p / Σ book` ratio.  With
     opposite-sign partition claims on one name (`Σ book ≈ 0`) the signed ratio exploded into
     a phantom large short, which later triggered a huge correction order (one 0.70-NAV SMCI
     buy → gross 1.44, cash −18 %, unwound the next day).  `BacktestExecutor` also has a
     **long-only backstop** (`long_only_tickers`, CN) so a sell can never take a restricted
     name net short from rounding dust.
  6. **Closed-market tradability mask + account caps** (2026-09-18): `BacktestData.tradeable(row)`
     requires a finite close today **and** finite open tomorrow; the strategy's mask is
     `window_valid & tradeable`, and the due partition **carries non-tradable holdings** instead
     of selling them.  Without this, CN names were allocated during CN Golden Week (US-only rows
     in the union calendar) and every fill was silently skipped → half the reallocation dropped,
     books diverged, and the partition attribution blew up (`nav_p → 0`, gross/nav → 9.8, tile
     gap → 1.92) exactly on the early-October holiday windows.  Also: account-level caps
     `gross ≤ max_gross_exposure` / `net ≤ max_net_exposure` (both default 1.0, so cash ≥ 0 —
     no borrowing), an executor **no-borrow backstop** (`no_borrow`, skips unaffordable buys),
     a `prev_weight_clip` on the turnover reference, and new diagnostics
     (`cash_unengaged`, `partition_tile_gap`, `neg_cash_days`, `over_gross_days`).
     Test count now **35** (3 test files); run commands in `README.md`.
- **Run commands** for the full-range 3/10/20 bp configs now live in `README.md` (§Venue 0
  backtest — run commands).
- **3-way checkpoint pre-selection RUNNING** (screen `presel`, 2025-01-02 → 2026-08-10,
  zero-cost): base stage-2 (`outputs/2026-08-15/20-05-06-814233/custom_tactis_medium.pth`) →
  RL 09-10 best → RL 09-14 best (`rl_step003504` / `rl_step000438`).  Output dirs
  `outputs/venv0_presel_{base,rl0910,rl0914}`; ~46 min each; results supersede all earlier
  Venue 0 numbers.
- **Post-fix zero-cost reference (09-14 best, pre-CN-ban)**: final NAV 1.249 (+24.9 % vs
  equal-weight universe +38.4 %), daily Sharpe **0.101** (Duke metric), annualized vol 8.1 %,
  max drawdown −5.9 %, avg turnover 0.32/day, gross ≤ 1.70 (one day), cash ≥ +0.14, partition
  tile = 5.9e-9.  (CN-ban pre-selection will revise these.)
- **Next**: pick the winner → **full-range run** (2008-01-02 → 2026-08-10, 4,832 trading days
  ≈ 9–10 h GPU) → fee ladder (CN 3 bp → 1 %) + `turnover_coef` sweep once "it looks good".

#### Locked decisions

| Decision | Choice |
|---|---|
| Strategy | **Span-forecast allocation** (no mode flag): N partitions (default = `pred_len` = 4), staggered daily rolls (exactly 1/N of the book re-allocated per day). The allocator optimizes the **4-day compounded span** `∏(1+r_h) − 1`, so the optimization target matches each partition's holding period. Buy-and-hold drift between rolls |
| Execution | Signal at close(t) → **fills at open(t+1)** (model runs off-hours; open prices exist in `all_stocks.csv`) |
| Costs | `commission_bps=0`, `slippage_bps=0` defaults, configurable |
| Turnover penalty | **ON by default**: `var_deployment.turnover_coef > 0` × `commission_rate` (floor 3e-4) — effective term `τ′·c·\|w − w_prev\|₁` (`portfolio/var_deployment.py:198`) with `prev_weights` = the rolling partition's reconciled weights (incl. its *real* attributed cash). The coef scales with the actual fee so the optimizer *anticipates* commissions (`τ′=16.7` → eff. ≈ 0.005 @ 3 bp, ≈ 0.167 @ 1 %). Intentionally inert in eval/val/RL (no persistent account → `prev_weights` undefined). The backtest is its first real consumer |
| Partition share reset | `share_reset: at_roll` — a partition's NAV share resets to 1/N of *current* NAV at each roll (trims winners / tops up losers at zero extra turnover, since it coincides with the roll trades anyway) |
| Partition history | **Fine-grained, persisted to CSV**: each partition keeps its own cash float + full history (per-day snapshots; per-roll records carrying the **full per-ticker weight vector**), appended iteratively to `partition_{p}_history.csv` under `outputs/.../ledger/`. RAM holds only live state; disk accumulates history (shared durable ledger for live venues later) |
| Layout | `trading/` package + top-level `backtest.py` runner |
| First run | Zero-cost baseline on 2025-01 → 2026-08-10 (415 days, ≈46 min/checkpoint) done; 3-way checkpoint pre-selection running; **full-range run (4,832 days ≈ 9–10 h GPU)** next, then the fee ladder |

#### Hard constraints

- Data extended to **2026-08-10** (the 2025–26 rows came from `data/custom-data-test/`; the
  single-ticker `sh.600000` tail past 08-10 was trimmed out).  Future re-extension uses the
  `--update` incremental mode in `download_us.py`/`download_cn.py` (fetch only dates after each
  ticker's last row, append, dedupe), then `merge.py`. **Universe frozen to the 1,302 base
  tickers** (model embedding size) — post-2024 IPOs (FDXF, SATS, WBA) are excluded from
  the backtest; alphabetical ticker order is preserved by `_pivot`.
- **Normalization frozen** to the 2008–2024 train region: first 80 % of the base 4,416
  timesteps (= 3,532). Extending the timeline must NOT shift norm stats —
  `CustomReturnDataset`'s first-80 % rule cannot be reused as-is; `trading/data.py`
  computes stats on the pinned base index range instead.
- Allocator band/threshold params (`return_lower/upper`, `max_loss`) are calibrated for
  **daily** returns; a 4-day span is ~2× daily vol → `backtest.allocators` carries
  span-tuned values (≈ ×2 vs daily defaults).
- Backtest calendar = sorted union of dates in `all_stocks.csv` (CN/US sessions share a
  date index; CN session t closes ~12 h before US session t). Suspended / missing-price
  names are frozen at last mark, not traded.

#### Files

```
trading/
  __init__.py
  base.py            # ExecutorBase ABC (shared book + diagnostics), PositionManager, Order/FillReport
  data.py            # BacktestData: date-indexed open/close/returns, frozen universe + norm stats
  strategy.py        # RollingPartitionStrategy (span samples, due-partition roll over list[Partition])
  portfolio.py       # Partition: per-partition cash/NAV book + fine-grained snapshots/roll records
  ledger.py          # CsvLedger: append-only durable CSV history (shared by all venues)
  samplers.py        # ModelSpanSampler (real) + RandomSpanSampler (mock)
  venues/
    __init__.py
    backtest.py      # BacktestExecutor: open(t+1) fills, shares/cash book, NAV marking, costs
backtest.py          # Hydra-style runner (mirrors eval.py: _load_config, stage-2 ckpt init)
tests/test_backtest_accounting.py
tests/test_partition_accounting.py
```

`ExecutorBase`: `get_holdings() -> dict[str, float]`, `get_shares()`, `get_nav()`,
`execute(orders, prices) -> FillReport`, `mark(prices) -> nav`, plus shared diagnostics
(`gross_exposure`, `net_exposure`, `cash_weight`) computed from the shared book.  Shared
book state (`shares`, `cash`, `last_prices`, `last_nav`, `bought_today`) lives in
`ExecutorBase.__init__` so `PositionManager` filters work venue-agnostically.
`PositionManager.plan(target)`: diff vs current holdings → risk filters (missing price,
per-market `allow_short`, CN T+1 same-session sells, `min_order_weight`) → returns orders;
`PositionManager.fill` is the executor's `execute` at the venue's chosen prices. IBKR/Xueqiu
executors plug into the same interface later.

**Per-partition history** (`trading/portfolio.py` + `trading/ledger.py`): each
`Partition` holds live state (`_dollars`, `_cash`, `_nav`; cash = residual so
`dollars.sum() + cash == nav` exactly — short proceeds accrue to the partition's cash)
and writes fine-grained history to an append-only CSV (`partition_{p}_history.csv`):
- `snapshot` rows per day: nav, dollars, cash, gross/net, period-return-since-roll.
- `roll` rows per re-allocation: nav-before/after reset, period return, cost, and the
  **full per-ticker weight vector** (JSON) — any partition's exact holdings on any date
  can be reconstructed.  Roll period returns give the per-holding-period return series
  (with `at_roll` each partition resets to `NAV/N`, so "partition gain/loss" is a rolling
  period-return series; `cumulative_return()` chains the rolls).
- `CsvLedger` appends and flushes per row → crash-safe and RAM-light; restart recovery
  re-reads the tail.  This is the shared durable ledger live venues (IBKR/Xueqiu) will use.
- Fill costs are attributed to the rolling partition's cash (`charge_due_costs`).

#### Strategy mechanics

```
Day t (after close): partition p due if (t − t0) mod N == p
  0. reconcile every partition book to the account's ACTUAL position dollars
     and cash (names not held are zeroed; held names distributed by book share;
     cash split by the SUM of partition grosses) → Σ partition NAVs == account NAV
  1. model.sample(hist through t) → [1, series, pred_len, n_samples]
     (TACTiS samples are joint across series AND time → the compound is a legit path draw)
  2. de-normalize each horizon step (z-score → raw daily returns)
  3. span = ∏_{h=1..4}(1 + r_h) − 1 per path → [1, series, n_samples]
     (3-line transform; feeds existing allocators unchanged)
  4. _augment_with_cash → allocator.allocate(..., prev_weights=partition reconciled weights)
  5. account-level target = account holdings + (new_dollars − reconciled book)/NAV
     (only the due partition changes → zero diff elsewhere → turnover ≈ 1/N)
```

Day loop: close(t) signal → open(t+1) fills (± slippage, commission) → close(t+1) mark → NAV.
Shares-based book; NAV = cash + Σ shares × close.

**Why reconciliation (bug fixed 2026-09-18):** the first Venue 0 runs let each partition's
internal dollar book drift from the executor's actual holdings (skipped fills on missing
prices, open-vs-close gaps, reset re-attribution). Emitted deltas then chased a phantom
book, and the account silently accumulated **unbounded leverage** (gross up to 9.34×, cash
−5.99×, partition NAVs not tiling). `reconcile()` grounds the books in the account every
day; a `gross_exposure > 1.5` warning guards the runner. Post-fix smoke: gross ≤ 0.66,
cash ≥ 0, partition NAVs tile exactly (tile = 0.0).

#### Config (implemented, `backtest:` section in `cfg/config.yaml`)

```yaml
backtest:
  seed: 42
  checkpoint: null            # required (ignored when mock_model=true)
  start_date: "2025-01-02"
  end_date: "2026-09-01"
  base_end_date: "2024-12-31" # universe + norm stats frozen to <= this date
  n_partitions: 4             # ≤ pred_len (hard limit: no forecast beyond pred_len)
  n_samples: 10
  execution_price: next_open  # next_open | next_close
  commission_bps: 0.0
  slippage_bps: 0.0
  allow_short: true
  share_reset: at_roll        # none | at_roll
  mock_model: false           # dev-only: random-span sampler, no GPU/model
  var_alpha: 0.05
  output_dir: null            # defaults to <checkpoint_dir>/backtest
  allocator:                  # same format as eval.allocators; span-tuned bands (≈×2 daily)
    name: var_deployment
    turnover_coef: 16.7        # τ′ × max(commission_rate, 3e-4) ≈ 0.005 @ 3 bp
    commission_rate: 0.0003  # per-trade fee fraction (3 bp CN); penalty floor 3e-4
    return_lower: -0.06       # ×2 of the -0.03 daily eval default
    return_upper: 0.12        # ×2 of the 0.06 daily eval default
    ...
```

`turnover_coef` scale: with `max_assets=50` and ±0.05 bounds, max possible turnover ≈ 5.0,
so the effective coef `τ′·c` at a few e-3 to e-2 produces penalties comparable to μ magnitudes
(~0.01).  With costs off the penalty only smooths churn; it becomes economically meaningful
once costs are on (the coef scales with `commission_rate`), so a single sweep over
`turnover_coef ∈ {0, 8, 16.7, 33}` covers both cost regimes (allocation is cheap; sampling
is shared).

#### Outputs

- `nav.csv` (date, weekday, nav, cash_weight, gross/net exposure, turnover, costs, skip
  reasons, per-partition nav/gross/period-return columns), `trades.csv`,
  `backtest_metrics.json` (incl. `partition_{p}/{n_rolls,total_return,final_nav,cost_total}`).
- **`ledger/partition_{p}_history.csv`** — append-only fine-grained per-partition history
  (per-day snapshot rows + per-roll rows with the full per-ticker weight vector as JSON),
  written iteratively during the run (see above).
- Headline metric: **competition-style daily Sharpe of NAV log returns (no annualization)** — identical to Duke FinTech scoring. Plus annualized Sharpe, total return, max drawdown, turnover/day, cost total, per-partition contribution.
- Benchmarks: equal-weight universe buy-and-hold (+ SPY buy-and-hold if in universe).
- The per-day attribution log enables post-hoc day-of-week / stop-loss analysis (see roadmap).

#### Correctness invariants (pytest)

NAV identity at every close (cash + Σ shares × price); account weights sum to 1 (gross + cash);
non-due partitions emit zero orders; turnover ≈ 1/N per day; no look-ahead (signal uses data
≤ t, fills use t+1 prices); partition NAV identity (dollars + cash == nav) after drift/roll;
cost attribution; Σ partition NAVs == account NAV (reconcile, tile = 0.0); gross bounded
(≤ ~1.0, the allocator's cap); cash never materially negative; no leverage accumulation;
ledger append/reopen/replay (crash-resume).  24 tests across `tests/test_backtest_accounting.py`,
`tests/test_partition_accounting.py`, `tests/test_subset_sampling.py`.

#### Optional strategy modules — roadmap

**v1 — base + cheap, all config-toggleable, all logged:**
- Rolling partitions + span allocator + `share_reset: at_roll` + per-day attribution log
  + **fine-grained per-partition CSV history** (snapshots + per-roll full weight vectors) — **implemented**
- Turnover penalty wired and ON by default (`backtest.allocator.turnover_coef=16.7` × `commission_rate=3e-4` ≈ eff. 0.005) — **implemented**
- CN/US market rule table in PositionManager: `cn {allow_short: false, t_plus_1: true, price_limit: 0.10}`, `us {allow_short: true}`. T+1 means a CN sell order can only reference shares held before today's open — **implemented** (`trading/base.py`). CN shorting is also banned *inside the allocator* (`var_deployment.long_only_mask` set from `BacktestData.market_mask("cn")`) so the optimizer's risk model (CVaR/band/beta) matches the book that actually executes, instead of planning phantom CN shorts that the PositionManager clamps away — **implemented** (`portfolio/var_deployment.py`)
- Vol targeting: scale gross so predicted book vol (std of the MC sample paths of the allocated book — available for free) ≈ target; generalizes `var_deployment`'s hard band. Toggle, default off until the baseline is measured — **not yet implemented**

**v1.5 — built toggled-off, enabled only if the attribution log justifies (measurement-first, avoid overfitting):**
- Day-of-week skip (`skip_weekdays: [...]` → due partition goes to cash instead of rolling). DOW buckets are ~85 days each in a 425-day window — noisy; tune on 2025, validate on 2026-H1; apparent DOW effects may be CN/US calendar artifacts
- Realized partition stop-loss (liquidate a partition to cash on X % drawdown since its roll; re-enter at reduced size) and/or account-level CPPI floor (exposure = multiplier × (NAV − floor)) — implement both, let the backtest judge
- Signal-strength gating: scale partition gross by cross-sectional dispersion of predicted span returns (no ranking edge → more cash)

**v2 — later:**
- CN-after-US conditioning: US close(t) is known ~12 h before CN open(t+1). Free version: importance-reweight the joint MC paths by proximity of their US component to the realized US outcome, then re-optimize only the CN weights (no re-inference — the copula already encodes cross-market dependence). Full version: two-phase rolls (phase 1 at US close for US/global sleeves, phase 2 at CN open re-conditioning the CN sleeve with US close(t) in history). Requires per-market roll phases and fills at different opens
- **Options overlay — monetize predicted vol instead of parking the hard band's cash.**
  Options exist per-listed-instrument, not per-portfolio, so a "portfolio vol position" is
  built by **factor decomposition**: index/sector-ETF straddle baskets (SPY/QQQ for the beta
  component, sector ETFs for sleeves) matching the book's exposures — the systematic part of
  the book's vol; a diversified ~50-name residual is mostly idiosyncratic anyway. Two regimes,
  decided by the model's predicted book vol σ̂ (std of the MC sample paths of the allocated
  book — free from sampling) vs implied vol:
  - **σ̂ > implied** (vol underpriced by the market): **buy** straddles/strangles. Lose the
    premium in the quiet range; profit when the market exceeds the breakeven interval in
    *either* direction. This is the direction-agnostic "high vol both ways" structure.
  - **σ̂ < implied** (vol overpriced): **sell** strangles/condors (vol-risk-premium harvest).
  Tenor matters: the prediction window is 4 days, so the implied reference is **VIX9D**
  (30-day VIX is the wrong horizon). **US sleeve only** — CN names (sz./sh.) have no listed
  options (A-share options need CN exchange access; US-listed China ETFs are a weak proxy).
  The hard band's de-levered cash is the premium budget / short-vol margin, so the band and
  the overlay compose. Buy-side vol has *negative unconditional expectancy* (VRP), so
  long-vol only fires on σ̂ − implied above a threshold margin — and only after the Phase-A
  spread log (σ̂ and VIX9D logged every day in the attribution log; no trading) shows the
  conditional edge exists. Feasibility ladder: Phase A log-only → Phase B simulated BSM fills
  (weekly rolls + costs) → Phase C live. Model-native stretch goal: the copula forecasts the
  dependence structure, so a predicted *dispersion* vs implied-correlation signal could drive
  the classic dispersion trade (sell index vol / buy single-name vol) — later, not the entry
  point.

**Rejected for now:** take-profit / gain capping — the edge is cross-sectional ranking, and the right tail of partition returns pays for the left; a symmetric stop/tp pair converts the return distribution into a barrier bet. Revisit only if the attribution log shows partition-level mean reversion after large gains.

### Duke FinTech (IBKR)


Context: this repo's model is being adapted to trade the
[Duke FinTech Trading Competition](https://fintechtradingcompetition.com/)
(paper trading on Interactive Brokers with simulated money). Rolling-entry, open to
all (not just university students), 3-month windows. Team = Yitao + agent.

#### Scoring — this repo's Sharpe IS the competition metric

The competition ranks traders on **risk-adjusted daily return**, no annualization:

```
Sharpe = (R_p − r_f) / σ_p
  R_p = geometric mean of daily log returns of the account NAV
  r_f = 3-month CMT (fixed at competition start, ~risk-free)
  σ_p = std.dev. of daily log returns of the account NAV
```

- Daily log returns of end-of-day NAV; first day has no return.
- **No sqrt(252)** — this is exactly the "raw per-period (daily) Sharpe" that
  `metrics.py`/`eval.py`/checkpoint names already use. Do NOT annualize.
- Lower volatility is rewarded: they explicitly penalize risk, so the
  `var_deployment` / `mv_long` variance + CVaR + beta controls are the right axis.
- Same fixed `r_f` for everyone for the whole window.

#### Current offline status (2026-09-18)

- **RL post-training DONE** (runs 09-10 / 09-12 / 09-14; best by test eval: RL 09-14
  `outputs/2026-09-14/11-04-54-235706/checkpoints/rl_step000438_sharpe0.0491.pt`).
- **Test evals**: RL 09-14 best recall@10_q95 = 10.3 % (var_deployment Sharpe 0.0604),
  RL 09-10 best 9.0 % (0.0377); base stage-2 has no recent eval (see the 3-way pre-selection).
- **Venue 0 pre-selection RUNNING** (screen `presel`, WSL root): base → RL 09-10 → RL 09-14,
  2025-01-02 → 2026-08-10, zero-cost, with reconciliation fix + CN short ban + τ′·c penalty.
- The job lives inside WSL root's `screen`, NOT on the Windows process table. Always inspect via:
  ```
  wsl -u root -d Ubuntu screen -ls
  wsl -u root -d Ubuntu ps aux | grep -E 'presel|backtest.py'
  wsl -u root -d Ubuntu tail -20 /root/venv0_preselect.log
  ```
  (`wsl -u root -d ubuntu` is required — the default user won't see root's screen.)

#### Target production architecture

```
EC2 (persistent)                  local workstation / WSL (on-demand, WOL to wake)
┌─────────────────────┐          ┌─────────────────────────────┐
│ IB Gateway headless │          │ TACTiS inference + RL train │
│ ib_insync client    │          │ HF_ENDPOINT=hf-mirror.com   │
│ daily EOD scheduler →───(ssh)→│ (HF Hub blocked from CN)    │
│ report/order flow   │          │ data sync EC2 ↔ workstation     │
└─────────────────────┘          └─────────────────────────────┘
```

- **EC2 = always-on trading node**: headless IB Gateway (Linux), `ib_insync` to
  place orders, cron for end-of-day scheduling.
- **workstation = compute on demand**: WOL to wake, run TACTiS inference / RL training,
  send target weights back. HF mirror env var required.
- Data & weights sync between EC2 and the workstation via a reliable path (rsync / shared NAS).

#### TODO / decision points (draft, to be refined)

1. Confirm competition sign-up status + exact Fall 2026 window start date.
2. Confirm IBKR paper account + where IB Gateway is installed (EC2 containerized?).
3. Choose instrument universe — closest to current pipeline (daily OHLCV, US tickers)
   is a basket of US ETFs / indices (fallback if the 1,302-stock book is unwieldy).
   Options overlay (see Venue 0 v2) is an optional later layer, not excluded.
4. Decide which allocator to deploy — don't wait for full 24-epoch RL if comp is near;
   use current best checkpoint (e.g. `rl_step000657_sharpe0.0383` or a mv_long run).
5. Commit + push untracked files (`portfolio/var_deployment.py`, `data/`, etc.) before
   relying on the repo.
6. Strategy is DAILY rebalance, low turnover, beta-controlled — do not chase high frequency.


### 雪球组合 (Xueqiu portfolio)

A Xueqiu portfolio (雪球组合) is a simulated/paper portfolio that can be
programmatically rebalanced. Cheap, low-friction long-only venue, separate from
the IBKR competition entry. Shares the SAME PositionManager layer — only the
executor differs.

#### Markets & constraints

- **Supported**: 沪深 A-shares, HK, US. A portfolio is built in ONE market type
  (`portfolio_market` in ['cn', 'us', 'hk']); cross-market portfolio not supported
  (模拟盈亏 may hold cross-market). Global coverage = one Xueqiu portfolio per market.
- **Long-only**: weights are 0-100%, no negative weight ⇒ no shorting. `mv_long`
  maps cleanly; `var_deployment` negatives must be renormalized to pure long for
  this venue (or run long-only here and short on IBKR).
- <=30 holdings per portfolio; single asset min weight 1% (can be a money-market
  fund like 华宝添益/银华日利).
- HK: stocks < HK$1 cannot be traded. Suspended/limit-D/delisted stocks rejected
  (stock `flag != 1`).

#### Auth & API (from easytrader, actively maintained)

- Auth = browser **cookies** (not password); store chmod 600, never in chat.
- Read current positions: `GET cubes/rebalancing/current.json?cube_symbol=<ZH>`
- Quote / net value: `GET cubes/quote.json`
- Stock lookup (need `stock_id`): `GET stock/p/search.json`
- **Rebalance = FULL rebalance**: `POST cubes/rebalancing/create.json` with
  `{cash, holdings: <json array>, cube_symbol, segment, comment}`. You submit the
  ENTIRE target holding structure and Xueqiu adjusts to it — no per-tick fills.
  Maps straight onto allocator target weights.
- History: `GET cubes/rebalancing/history.json`
- Convenience wrapper: `easytrader.XueQiuTrader` (adjust_weight, get_position,
  get_balance, history, _trade). Only ~3-4 requests to reimplement if preferred.

#### Scheduling & cadence

- Daily full rebalance EOD, slow/rate-limited (browser-cookie auth is fragile).
- Looks like a paper beauty-contest mirror of the IBKR entry; use for A-share/HK
  long-only track or double-run validation, not as the primary shorting venue.

### Venues 1 & 2 — Live executors (implemented 2026-09-20)

Status: the two live executors, the shared safety layer, config and offline tests are
**implemented**.  A live runner (`run_live.py`) is deliberately deferred — this pass
builds the executors + dry-run + tests only.  `pytest tests/` runs offline (71 tests);
no `ib_async`/`easytrader` import is required for tests.

#### Safety model (two-factor, shared)

`trading/live.py` → `LiveGuard`.  An order may leave the machine **only if all hold**:
1. the venue block's `live: true`, **and**
2. env `TACTIS_LIVE=1`.

Otherwise it is forced dry-run and reasons are surfaced in `FillReport.skipped`
(`dry_run`, `live_disabled`, `no_live_env`).  A **kill-switch** (env
`TACTIS_TRADING_KILL=1`, or a sentinel file `live.kill_switch_file`) and a per-venue
**notional cap** (`max_notional`) apply on top.  Local is dev-only; live runs on EC2
later (IB Gateway paper = port 4002; Xueqiu cookies provisioned there only).

#### Config — `cfg/application.yaml` (separate from `cfg/config.yaml`)

Holds `live:`, `ibkr:`, `xueqiu:` with safe defaults (`live: false`, `dry_run: true`).
Loaded via `trading.live.load_application_config()`.  Training/backtest configs are
untouched by live settings.

#### Venue 1 — IBKR (`trading/venues/ibkr.py`)

- `IBKRBroker` (lazy `ib_async`, `ib_insync` fallback) + `IBKRExecutor(ExecutorBase)`.
- **Markets**: US (long+short) and China A-shares via Stock Connect (long-only).
  A per-market `markets` table drives contracts, lot size and shortability:
  - US → `Contract(symbol, STK, SMART, USD)` with `symbol_replace={".": " "}`
    (BRK.B → BRK B), whole shares, shortable;
  - CN (`sh.`/`sz.`) → symbol strips the prefix (`sh.600000` → `600000`), exchange
    `SEHKNTL`/`SEHKSZSE`, currency `CNH`, **100-share board lots** (buys floored),
    **long-only** (Stock Connect offers no A-share shorting).
  `contract_map` overrides any ticker.  `sync()` reads positions/NAV from the broker
  (source of truth) and maps IB symbols back (6-digit codes → `sh.`/`sz.` by range).
- `execute()`: diff → board-lot rounding → `LiveGuard` → dry-run (log only, **no book
  mutation**) / live `MOO`+`tif=OPG` orders.  Reuses `trading/costs.build_cost_model`
  (IBKR schedule) for the pre-trade estimate.
- Offline invariants: dry-run submits nothing; `live` without env blocks; kill-switch;
  notional cap; board-lot rounding (`rounds_to_zero`); per-market long-only clamp;
  CN/US mixed batch; sync mapping.
- **Data source**: the model still reads offline `all_stocks.csv` (yfinance/baostock);
  the executor does **not** request market data from IBKR — prices are passed in by the
  (deferred) runner.  Accepted for now.

#### Venue 2 — Xueqiu (`trading/venues/xueqiu.py`)

- Xueqiu is **target-based** (full rebalance), not delta/fill-based.  Primary path is
  venue-native `XueqiuExecutor.rebalance(target)`; `execute(orders, prices)` is a shim
  that reconstructs the target from the order diff and delegates, so the shared
  `ExecutorBase` contract still holds.
- `XueqiuClient` wraps `easytrader` behind an injectable transport (cookie from env or a
  chmod-600 file; throttled; retrying; the cookie is never logged).  `easytrader` is an
  applications-only dependency (`requirements-app.txt`), so tests inject a fake client.
- CN long-only projection before submission: drop negatives, ≤30 holdings, ≥1 % per name,
  cash residual optionally parked in `cash_proxy_ticker` (money-market fund).
- Offline invariants: projection (long-only / min-weight / top-30 / cash proxy);
  dry-run posts nothing; live posts one payload within limits; symbol normalisation
  (`sh.600000` ↔ `SH600000`); throttle / retry / cookie handling.

#### Files & deps

```
NEW  trading/costs.py            # CostModel / MarketCost / build_cost_model (shared)
MOD  trading/venues/backtest.py  # re-exports CostModel/MarketCost from ..costs
NEW  trading/live.py             # LiveGuard + load_application_config
NEW  trading/venues/ibkr.py      # IBKRBroker + IBKRExecutor
NEW  trading/venues/xueqiu.py    # XueqiuClient + XueqiuExecutor
NEW  cfg/application.yaml        # live:/ibkr:/xueqiu:
NEW  requirements-app.txt        # ib_async, easytrader
NEW  tests/test_live_guards.py, tests/test_ibkr_venue.py, tests/test_xueqiu_venue.py
```

#### Commands

```bash
# offline tests for the live layer (no broker, no network, no cookie)
pytest tests/test_live_guards.py tests/test_ibkr_venue.py tests/test_xueqiu_venue.py -v

# install the application-only dependencies (live machines only)
pip install -r requirements-app.txt
```

To enable live on a venue: set the venue block's `live: true` and `dry_run: false` in
`cfg/application.yaml`, export `TACTIS_LIVE=1` (and *not* `TACTIS_TRADING_KILL=1`).
Without all of these the executor stays dry-run.
