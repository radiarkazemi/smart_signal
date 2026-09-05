from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """TFT-style gated residual block."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_hidden, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        y = self.fc2(h)
        g = torch.sigmoid(self.gate(h))
        return self.norm(self.skip(x) + self.drop(g * y))


class VariableSelectionNetwork(nn.Module):
    """Soft feature selection per timestep (vectorized TFT-style VSN)."""

    def __init__(self, n_features: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.feat_proj = nn.Linear(1, d_model)
        self.weight_grn = GatedResidualNetwork(n_features, d_model, n_features, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        weights = torch.softmax(self.weight_grn(x), dim=-1)
        transformed = self.feat_proj(x.unsqueeze(-1))  # (B, T, F, D)
        out = torch.sum(transformed * weights.unsqueeze(-1), dim=-2)
        return self.drop(out), weights


class CausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        y = self.drop(F.gelu(self.norm1(y)))
        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = self.drop(F.gelu(self.norm2(y)))
        return x + y


class TimeframeEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_tcn: int,
        kernel: int,
        n_tf_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
        self.tcn = nn.ModuleList(
            [TCNBlock(d_model, kernel, dilation=2**i, dropout=dropout) for i in range(n_tcn)]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_tf_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, weights = self.vsn(x)
        for block in self.tcn:
            h = block(h)
        h = self.transformer(h)
        return self.out_norm(h), weights


class CrossTimeframeFusion(nn.Module):
    """Fast timeframe queries slower timeframes (HTF directional bias)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_1h = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_4h = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_1d = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mix = GatedResidualNetwork(d_model * 4, d_model * 2, d_model, dropout)
        self.pool = nn.Linear(d_model, 1)

    def _pool(self, seq: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.pool(seq).squeeze(-1), dim=1)
        return torch.sum(seq * w.unsqueeze(-1), dim=1)

    def forward(
        self,
        h_15m: torch.Tensor,
        h_1h: torch.Tensor,
        h_4h: torch.Tensor,
        h_1d: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        c1h, w1h = self.attn_1h(h_15m, h_1h, h_1h, need_weights=True, average_attn_weights=True)
        c4h, w4h = self.attn_4h(h_15m, h_4h, h_4h, need_weights=True, average_attn_weights=True)
        c1d, w1d = self.attn_1d(h_15m, h_1d, h_1d, need_weights=True, average_attn_weights=True)
        fused_seq = h_15m + 0.5 * (c1h + c4h + c1d)
        pooled = torch.cat(
            [self._pool(h_15m), self._pool(c1h), self._pool(c4h), self._pool(c1d)],
            dim=-1,
        )
        fused = self.mix(pooled)
        attn = {"1h": w1h.detach(), "4h": w4h.detach(), "1d": w1d.detach()}
        return fused, attn | {"seq": fused_seq}


class GoldNet(nn.Module):
    """Multi-timeframe TCN + Transformer + hierarchical cross-attention.

    Heads:
      - direction: sell / hold / buy
      - expected log-return
      - realized |return| (volatility / move size)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        d_hidden: int = 128,
        n_heads: int = 4,
        n_tcn_layers: int = 4,
        tcn_kernel: int = 3,
        n_transformer_layers: int = 2,
        dropout: float = 0.18,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        enc_kwargs = dict(
            n_features=n_features,
            d_model=d_model,
            n_tcn=n_tcn_layers,
            kernel=tcn_kernel,
            n_tf_layers=n_transformer_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.enc_15m = TimeframeEncoder(**enc_kwargs)
        self.enc_1h = TimeframeEncoder(**enc_kwargs)
        self.enc_4h = TimeframeEncoder(**enc_kwargs)
        self.enc_1d = TimeframeEncoder(**enc_kwargs)
        self.fusion = CrossTimeframeFusion(d_model, n_heads, dropout)
        self.regime = nn.GRU(d_model, d_model, num_layers=gru_layers, batch_first=True, dropout=0.0)
        self.regime_norm = nn.LayerNorm(d_model)
        self.head_in = GatedResidualNetwork(d_model * 2, d_hidden, d_hidden, dropout)
        self.dir_head = nn.Linear(d_hidden, 3)
        self.ret_head = nn.Linear(d_hidden, 1)
        self.vol_head = nn.Linear(d_hidden, 1)
        # Next-candle teaching heads: bear/flat/bull + OHLC path targets.
        self.candle_head = nn.Linear(d_hidden, 3)
        self.next_high_head = nn.Linear(d_hidden, 1)
        self.next_low_head = nn.Linear(d_hidden, 1)
        self.next_close_head = nn.Linear(d_hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor], *, explain: bool = False) -> dict[str, torch.Tensor]:
        h15, w15 = self.enc_15m(batch["x_15m"])
        h1h, w1h = self.enc_1h(batch["x_1h"])
        h4h, w4h = self.enc_4h(batch["x_4h"])
        h1d, w1d = self.enc_1d(batch["x_1d"])
        fused, attn = self.fusion(h15, h1h, h4h, h1d)
        regime_seq, _ = self.regime(attn["seq"])
        regime = self.regime_norm(regime_seq[:, -1])
        h = self.head_in(torch.cat([fused, regime], dim=-1))
        out = {
            "dir_logits": self.dir_head(h),
            "y_ret": self.ret_head(h).squeeze(-1),
            "y_vol": F.softplus(self.vol_head(h).squeeze(-1)),
            "candle_logits": self.candle_head(h),
            "y_next_high": self.next_high_head(h).squeeze(-1),
            "y_next_low": self.next_low_head(h).squeeze(-1),
            "y_next_close": self.next_close_head(h).squeeze(-1),
        }
        if explain:
            out["vsn_15m"] = w15.mean(dim=1)
            out["vsn_1h"] = w1h.mean(dim=1)
            out["attn_1h"] = attn["1h"]
        return out


def build_goldnet(cfg: dict) -> GoldNet:
    from smart_signal.features.indicators import FEATURE_COLUMNS

    m = cfg.get("model") or {}
    feats = cfg.get("features") or FEATURE_COLUMNS
    n_features = len(feats) if feats else len(FEATURE_COLUMNS)
    return GoldNet(
        n_features=n_features,
        d_model=int(m.get("d_model", 64)),
        d_hidden=int(m.get("d_hidden", 128)),
        n_heads=int(m.get("n_heads", 4)),
        n_tcn_layers=int(m.get("n_tcn_layers", 4)),
        tcn_kernel=int(m.get("tcn_kernel", 3)),
        n_transformer_layers=int(m.get("n_transformer_layers", 2)),
        dropout=float(m.get("dropout", 0.18)),
        gru_layers=int(m.get("gru_layers", 1)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
