"""
TACTiS model wrapper with NaN-aware loss.

``TACTiSModel`` wraps the upstream ``TACTiS`` class, preserving its
interface (``loss``, ``sample``, ``set_stage``, ``initialize_stage2``) while
adding per-position NaN masking so that NaN-filled prediction timesteps do
not contribute to the marginal or copula loss.
"""

from contextlib import contextmanager

import torch
from torch import nn

from tactis.model.tactis import TACTiS


class _GatherSeriesEmbedding(nn.Module):
    """
    Stand-in for a ``series_encoder`` nn.Embedding that gathers a fixed subset.

    ``TACTiS.sample()``/loss look up per-series embeddings as
    ``series_encoder(torch.arange(num_series))``, i.e. they infer series
    identity purely from column order.  When forecasting only a random subset
    of series, the ``arange`` argument is meaningless: this wrapper ignores it
    and returns ``base(subset_idx)`` rows instead, so a sliced column block is
    scored under the *actual* subset's embeddings rather than sequential
    ``0..subset-1`` rows.
    """

    def __init__(self, base: nn.Module, subset_idx: list[int] | torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer(
            "_idx",
            torch.as_tensor(subset_idx, dtype=torch.long)
            if not isinstance(subset_idx, torch.Tensor)
            else subset_idx,
        )

    def forward(self, _indices_by_position: torch.Tensor) -> torch.Tensor:
        return self.base(self._idx.to(_indices_by_position.device))


@contextmanager
def subset_scope(models, subset_idx: list[int]):
    """
    Run ``sample``/``loss`` on only the given subset series, with correct identity.

    Swaps ``flow_series_encoder`` and ``copula_series_encoder`` on each model's
    underlying ``TACTiS`` instance for gather wrappers bound to ``subset_idx``,
    and restores the original modules on exit.  ``models`` may be a single
    ``TACTiSModel`` or an iterable of them.
    """
    if isinstance(models, nn.Module):
        models = [models]

    pairings = []
    for model in models:
        tact = model.tactis
        saved: dict[str, nn.Module] = {}
        for name in ("flow_series_encoder", "copula_series_encoder"):
            base = getattr(tact, name, None)
            if base is None:
                continue
            saved[name] = base
            setattr(tact, name, _GatherSeriesEmbedding(base, subset_idx))
        pairings.append((tact, saved))

    try:
        yield
    finally:
        for tact, saved in pairings:
            for name, base in saved.items():
                setattr(tact, name, base)


def build_model(cfg, num_series: int) -> "TACTiSModel":
    import inspect

    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.model, resolve=True)
    params.pop("name", None)
    params["num_series"] = num_series

    known = set(inspect.signature(TACTiS.__init__).parameters) - {"self"}
    unknown = sorted(set(params) - known)
    if unknown:
        raise TypeError(
            f"Unknown TACTiS config key(s) under cfg/model/{cfg.model.name}: "
            f"{', '.join(unknown)}. Valid keys: {sorted(known)}"
        )
    return TACTiSModel(params, device=torch.device(cfg.training.device))


class TACTiSModel(nn.Module):
    """
    Wrapper around ``TACTiS`` that adds NaN-aware loss masking.

    ``nan_pred_mask`` is a boolean tensor ``[batch, series, pred_len]`` where
    ``True`` means "this is a real prediction target" and ``False`` means
    "this position is NaN-filled, do not predict it".  Positions marked
    ``False`` are excluded from the decoder loss (both marginal and copula).

    Parameters
    ----------
    params : dict
        Keyword arguments forwarded to ``TACTiS.__init__``.
    device : torch.device
        Device to place the model on.
    """

    def __init__(self, params: dict, device: torch.device):
        super().__init__()
        self.tactis = TACTiS(**params).to(device)

    # ---- delegates -----------------------------------------------------------

    def parameters(self, recurse: bool = True):
        return self.tactis.parameters(recurse=recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ):
        return self.tactis.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.tactis.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.tactis.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        self.tactis.train(mode)
        return self

    def eval(self):
        self.tactis.eval()
        return self

    def to(self, *args, **kwargs):
        self.tactis.to(*args, **kwargs)
        return self

    def set_stage(self, stage: int):
        self.tactis.set_stage(stage)

    def initialize_stage2(self):
        self.tactis.initialize_stage2()

    @property
    def marginal_logdet(self):
        return self.tactis.marginal_logdet

    @property
    def copula_loss(self):
        return self.tactis.copula_loss

    # ---- core API ------------------------------------------------------------

    def loss(
        self,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: torch.Tensor,
        nan_pred_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute TACTiS loss with optional NaN-aware masking.

        Parameters
        ----------
        hist_time : Tensor  [batch, (series,) time]
            Time indices for history.
        hist_value : Tensor  [batch, series, H]
            Historical returns (NaN-filled with 0.0, normalized).
        pred_time : Tensor  [batch, (series,) time]
            Time indices for prediction.
        pred_value : Tensor  [batch, series, P]
            Prediction targets (NaN-filled with 0.0, normalized).
        nan_pred_mask : Tensor | None  [batch, series, P] bool
            ``True`` = real prediction target.  ``False`` = NaN-filled,
            excluded from loss computation.  ``None`` = predict all.

        Returns
        -------
        marginal_logdet : Tensor []
        copula_loss : Tensor []
        """
        m = self.tactis
        batch = hist_value.shape[0]
        ns = hist_value.shape[1]
        H = hist_value.shape[2]
        P = pred_value.shape[2]
        device = hist_value.device

        copula_se = None

        flow_se = m.flow_series_encoder(torch.arange(ns, device=device))
        if not m.skip_copula:
            copula_se = m.copula_series_encoder(torch.arange(ns, device=device))
        flow_se = flow_se[None, :, :].expand(batch, -1, -1)
        if not m.skip_copula:
            copula_se = copula_se[None, :, :].expand(batch, -1, -1)

        if hist_time.dim() == 2:
            hist_time = hist_time[:, None, :]
        if pred_time.dim() == 2:
            pred_time = pred_time[:, None, :]
        if hist_time.shape[1] == 1:
            hist_time = hist_time.expand(-1, ns, -1)
        if pred_time.shape[1] == 1:
            pred_time = pred_time.expand(-1, ns, -1)

        if m.bagging_size:
            bags = [torch.randperm(ns, device=device)[: m.bagging_size] for _ in range(batch)]
            hist_time = torch.stack([hist_time[i, bags[i], :] for i in range(batch)], dim=0)
            hist_value = torch.stack([hist_value[i, bags[i], :] for i in range(batch)], dim=0)
            pred_time = torch.stack([pred_time[i, bags[i], :] for i in range(batch)], dim=0)
            pred_value = torch.stack([pred_value[i, bags[i], :] for i in range(batch)], dim=0)
            flow_se = torch.stack([flow_se[i, bags[i], :] for i in range(batch)], dim=0)
            if not m.skip_copula:
                copula_se = torch.stack([copula_se[i, bags[i], :] for i in range(batch)], dim=0)
            if nan_pred_mask is not None:
                nan_pred_mask = torch.stack([nan_pred_mask[i, bags[i], :] for i in range(batch)], dim=0)
            ns = m.bagging_size

        normalizer = m.data_normalization(hist_value)
        hist_value = normalizer.normalize(hist_value)
        pred_value = normalizer.normalize(pred_value)

        true_value = torch.cat([hist_value, pred_value], dim=2)

        hist_mask = torch.ones(batch, ns, H, dtype=bool, device=device)
        pred_mask = torch.zeros(batch, ns, P, dtype=bool, device=device)

        # ---- NaN exclusion overlay ----
        if nan_pred_mask is not None:
            pred_mask = pred_mask | (~nan_pred_mask.bool())
        mask = torch.cat([hist_mask, pred_mask], dim=2)

        hist_enc_flow = torch.cat(
            [
                hist_value[:, :, :, None],
                flow_se[:, :, None, :].expand(batch, -1, H, -1),
                torch.ones(batch, ns, H, 1, device=device),
            ],
            dim=3,
        )
        pred_enc_flow = torch.cat(
            [
                torch.zeros(batch, ns, P, 1, device=device),
                flow_se[:, :, None, :].expand(batch, -1, P, -1),
                torch.zeros(batch, ns, P, 1, device=device),
            ],
            dim=3,
        )
        if not m.skip_copula:
            hist_enc_cop = torch.cat(
                [
                    hist_value[:, :, :, None],
                    copula_se[:, :, None, :].expand(batch, -1, H, -1),
                    torch.ones(batch, ns, H, 1, device=device),
                ],
                dim=3,
            )
            pred_enc_cop = torch.cat(
                [
                    torch.zeros(batch, ns, P, 1, device=device),
                    copula_se[:, :, None, :].expand(batch, -1, P, -1),
                    torch.zeros(batch, ns, P, 1, device=device),
                ],
                dim=3,
            )

        flow_enc = torch.cat([hist_enc_flow, pred_enc_flow], dim=2)
        flow_enc = m.flow_input_encoder(flow_enc)
        if not m.skip_copula:
            copula_enc = torch.cat([hist_enc_cop, pred_enc_cop], dim=2)
            copula_enc = m.copula_input_encoder(copula_enc)
        if m.input_encoding_normalization:
            flow_enc = flow_enc * (m.flow_encoder_embedding_dim**0.5)
            if not m.skip_copula:
                copula_enc = copula_enc * (m.copula_encoder_embedding_dim**0.5)

        timesteps = torch.cat([hist_time, pred_time], dim=2)
        flow_enc = m.flow_time_encoding(flow_enc, timesteps.to(int))
        if not m.skip_copula:
            copula_enc = m.copula_time_encoding(copula_enc, timesteps.to(int))

        flow_enc = m.flow_encoder.forward(flow_enc)
        if not m.skip_copula:
            copula_enc = m.copula_encoder.forward(copula_enc)
        else:
            copula_enc = None

        _ = m.decoder.loss(
            flow_encoded=flow_enc,
            copula_encoded=copula_enc,
            mask=mask,
            true_value=true_value,
        )

        m.copula_loss = m.decoder.copula_loss
        m.marginal_logdet = m.decoder.marginal_logdet

        if m.loss_normalization in {"series", "both"}:
            m.copula_loss = m.copula_loss / ns
            m.marginal_logdet = m.marginal_logdet / ns
        if m.loss_normalization in {"timesteps", "both"}:
            m.copula_loss = m.copula_loss / P
            m.marginal_logdet = m.marginal_logdet / P

        return m.marginal_logdet, m.copula_loss

    def sample(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
    ) -> torch.Tensor:
        return self.tactis.sample(
            num_samples=num_samples,
            hist_time=hist_time,
            hist_value=hist_value,
            pred_time=pred_time,
        )

    def sample_subset(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        series_idx: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Jointly sample from the distribution of only ``series_idx`` series.

        Slices ``hist_value`` to the requested series and runs the vanilla
        ``sample`` pipeline under a :func:`subset_scope`, so the forecast
        conditions only on the selected series' histories and is scored with
        their actual per-series embeddings.  Returns
        ``[batch, len(series_idx), hist_len + pred_len, num_samples]`` in
        ``series_idx`` column order.

        If ``hist_time``/``pred_time`` carry a per-series dimension (> 1), they
        are sliced to match; batch-singleton time tensors (as produced by
        ``make_time_tensors``) need no slicing.
        """
        hist_value = hist_value[:, series_idx, :]
        if hist_time.dim() == 3 and hist_time.shape[1] > 1:
            hist_time = hist_time[:, series_idx, :]
        if pred_time.dim() == 3 and pred_time.shape[1] > 1:
            pred_time = pred_time[:, series_idx, :]
        with subset_scope(self, series_idx):
            return self.tactis.sample(
                num_samples=num_samples,
                hist_time=hist_time,
                hist_value=hist_value,
                pred_time=pred_time,
            )
