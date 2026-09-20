"""
Span samplers for the rolling-partition strategy.

Both expose ``__call__(row) -> [1, S, n_samples]`` of *raw* (de-normalized)
compounded span returns, so the strategy never sees z-score space.
"""

import numpy as np
import torch

from trainer import make_time_tensors


class ModelSpanSampler:
    """
    Draw joint ``pred_len``-day samples from a TACTiS model and compound them
    into raw span returns.

    Parameters
    ----------
    model : TACTiSModel
        Trained model (stage 2 if the copula is active).
    data : BacktestData
        Supplies z-scored history windows and per-series norm stats.
    hist_len, pred_len : int
        Model window parameters.
    n_samples : int
        Monte-Carlo samples per call.
    device : torch.device
        Compute device (matches the model).
    """

    def __init__(self, model, data, hist_len: int, pred_len: int, n_samples: int, device):
        self.model = model
        self.data = data
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.n_samples = n_samples
        self.device = device
        self._means, self._stds = data.norm_tensors(device)

    def __call__(self, row: int) -> torch.Tensor:
        z = torch.from_numpy(self.data.z_hist(row, self.hist_len))[None].to(self.device)
        hist_time, pred_time = make_time_tensors(self.hist_len, self.pred_len, 1, self.device)
        with torch.no_grad():
            samples = self.model.sample(
                num_samples=self.n_samples,
                hist_time=hist_time,
                hist_value=z,
                pred_time=pred_time,
            )
        # samples: [1, S, hist_len + pred_len, n_samples] → take the prediction steps
        sp = samples[:, :, -self.pred_len :, :]                     # [1, S, P, n]
        raw = sp * self._stds[None, :, None, None] + self._means[None, :, None, None]
        span = (1.0 + raw).prod(dim=2) - 1.0                        # [1, S, n]
        return span


class RandomSpanSampler:
    """
    Deterministic Gaussian span sampler for correctness checks (no model/GPU).

    Produces centred draws with per-name scale ~ ``span_vol`` (roughly matching
    a ~4-day compounded daily vol of a few percent).  Used with ``mock_model``.
    """

    def __init__(
        self,
        n_series: int,
        n_samples: int,
        span_vol: float = 0.03,
        seed: int = 42,
        device: torch.device | None = None,
    ):
        self.n_series = n_series
        self.n_samples = n_samples
        self.span_vol = span_vol
        self.device = device or torch.device("cpu")
        self._rng = torch.Generator(self.device).manual_seed(seed)

    def __call__(self, _row: int) -> torch.Tensor:
        return torch.randn(1, self.n_series, self.n_samples, generator=self._rng,
                           device=self.device) * self.span_vol
