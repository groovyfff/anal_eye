"""PyTorch network definitions for the sequence layer.

Three interchangeable architectures (selectable via ``ModelConfig.sequence_backend``):

* ``LSTMNet`` / ``GRUNet`` - classic recurrent encoders.
* ``PatchTST`` - a patched Transformer encoder for time series (channel-
  independent patch embedding + Transformer encoder), faithful to the PatchTST
  design but compact enough for low-latency fp16 inference on a P100.

Each net consumes a window of shape ``(batch, seq_len, n_channels)`` and emits a
single logit for ``P(trend continuation)`` plus a scalar trend-sign head.

Torch is imported lazily at call-time so the rest of the package stays usable on
machines without a CUDA build.
"""

from __future__ import annotations

from typing import Any


def _torch():
    import torch  # noqa: WPS433

    return torch


def _nn():
    import torch.nn as nn  # noqa: WPS433

    return nn


def build_network(backend: str, n_channels: int, seq_len: int) -> Any:
    """Factory returning an initialized nn.Module for the chosen backend."""
    backend = backend.lower()
    if backend == "lstm":
        return LSTMNet(n_channels)
    if backend == "gru":
        return GRUNet(n_channels)
    if backend == "patchtst":
        return PatchTST(n_channels=n_channels, seq_len=seq_len)
    raise ValueError(f"unknown sequence backend {backend!r}")


def _make_lstm_net(cls_name: str):
    nn = _nn()
    torch = _torch()

    class _RecurrentHead(nn.Module):
        def __init__(self, n_channels: int, hidden: int = 128, layers: int = 2, gru: bool = False):
            super().__init__()
            rnn_cls = nn.GRU if gru else nn.LSTM
            self.rnn = rnn_cls(
                input_size=n_channels,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=0.1,
            )
            self.norm = nn.LayerNorm(hidden)
            self.cont_head = nn.Linear(hidden, 1)  # continuation logit
            self.sign_head = nn.Linear(hidden, 1)  # trend sign in [-1, 1] via tanh

        def forward(self, x):  # x: (B, T, C)
            out, _ = self.rnn(x)
            h = self.norm(out[:, -1, :])
            cont_logit = self.cont_head(h).squeeze(-1)
            sign = torch.tanh(self.sign_head(h)).squeeze(-1)
            return cont_logit, sign

    _RecurrentHead.__name__ = cls_name
    return _RecurrentHead


class _LazyNet:
    """Descriptor-free lazy wrappers so the module imports without torch."""


def LSTMNet(n_channels: int):  # noqa: N802 (factory style)
    cls = _make_lstm_net("LSTMNet")
    return cls(n_channels, gru=False)


def GRUNet(n_channels: int):  # noqa: N802
    cls = _make_lstm_net("GRUNet")
    return cls(n_channels, gru=True)


def PatchTST(
    n_channels: int,
    seq_len: int,
    patch_len: int = 8,
    stride: int = 4,
    dropout: float = 0.2,
    attn_dropout: float = 0.15,
):  # noqa: N802
    nn = _nn()
    torch = _torch()

    class _PatchTST(nn.Module):
        """Channel-independent patched Transformer for time series.

        Dropout is applied (a) inside the Transformer encoder and (b) on the
        flattened representation before the heads. ``nn.Dropout`` carries *no*
        parameters, so adding it does not change ``state_dict`` keys -- weights
        trained with this module load cleanly into a module without it (and
        vice-versa), and dropout is a no-op at eval time.
        """

        def __init__(self) -> None:
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride
            self.n_channels = n_channels
            self.d_model = 128
            self.n_patches = max(1, (seq_len - patch_len) // stride + 1)

            self.patch_embed = nn.Linear(patch_len, self.d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=256,
                dropout=attn_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.flatten = nn.Flatten(start_dim=1)
            self.head_dropout = nn.Dropout(dropout)
            feat_dim = self.d_model * self.n_patches * n_channels
            self.cont_head = nn.Linear(feat_dim, 1)
            self.sign_head = nn.Linear(feat_dim, 1)

        def _patchify(self, x):  # x: (B, T, C) -> (B*C, n_patches, patch_len)
            b, t, c = x.shape
            x = x.permute(0, 2, 1)  # (B, C, T)
            patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            # patches: (B, C, n_patches, patch_len)
            return patches.reshape(b * c, -1, self.patch_len), b, c

        def forward(self, x):
            patches, b, c = self._patchify(x)
            emb = self.patch_embed(patches) + self.pos_embed[:, : patches.size(1), :]
            enc = self.encoder(emb)  # (B*C, n_patches, d_model)
            enc = enc.reshape(b, c, enc.size(1), enc.size(2))
            feat = self.head_dropout(self.flatten(enc.reshape(b, -1)))
            cont_logit = self.cont_head(feat).squeeze(-1)
            sign = torch.tanh(self.sign_head(feat)).squeeze(-1)
            return cont_logit, sign

    return _PatchTST()
