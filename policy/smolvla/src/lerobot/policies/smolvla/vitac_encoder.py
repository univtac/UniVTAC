# Copyright 2026 The vitac_smolvla team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""ViTacEncoder: visuo-tactile encoder for SmolVLA.

Three integration paths are supported by the rest of the codebase:

1.  ``token_fusion`` -- tactile images are treated as additional visual
    observations (named ``observation.images.tactile_*``) and pass through
    SmolVLM's vision encoder along with the head/wrist cameras. In this mode
    the ViTacEncoder is *not* used at all; we only need the data to look like
    images.

2.  ``cross_attn`` -- tactile images are routed *outside* of SmolVLM and
    encoded by a :class:`ViTacEncoder` instance, then projected to the prefix
    embedding space by a :class:`TactileAlignmentHead` MLP. The resulting
    tokens are appended to the prefix as a separate attention block so that
    the action expert attends to them as cross-attention condition tokens.

3.  ``film`` -- tactile features are pooled, projected through
    :class:`TactileAlignmentHead`, and then fed to a :class:`FiLMHead` that
    produces feature-wise gamma/beta modulation parameters. These are applied
    to the action+time embedding inside the action expert at every denoising
    step, so tactile signal directly modulates the denoising trajectory.

The class hierarchy is intentionally a thin base + a concrete stub. Future
work can plug in stronger tactile backbones (e.g., GelSight-specific models,
marker-flow networks) by subclassing :class:`BaseViTacEncoder`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

log = logging.getLogger(__name__)


class BaseViTacEncoder(nn.Module, ABC):
    """Abstract base class for visuo-tactile encoders.

    A ViTacEncoder consumes one or more tactile image streams and produces a
    sequence of *raw* token embeddings shaped ``(B, num_tokens, feature_dim)``.
    Those raw tokens are *not* yet in the cross-attention space of the
    SmolVLA prefix; a downstream :class:`TactileAlignmentHead` is responsible
    for projecting them to the appropriate embedding dimension (text hidden
    size for cross-attention, expert hidden size for FiLM).

    Subclasses must implement :meth:`encode_one` which encodes a single
    tactile stream of shape ``(B, 3, H, W)``. The base class is responsible
    for orchestrating per-stream encoding and concatenating their outputs
    along the sequence dimension.
    """

    def __init__(self, feature_dim: int, num_tokens_per_stream: int = 8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tokens_per_stream = num_tokens_per_stream

    @abstractmethod
    def encode_one(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single tactile stream.

        Args:
            x: Image tensor of shape ``(B, 3, H, W)`` in ``[-1, 1]``.

        Returns:
            Token tensor of shape ``(B, num_tokens_per_stream, feature_dim)``.
        """

    def forward(
        self, tactile_images: Sequence[torch.Tensor], tactile_masks: Sequence[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an iterable of tactile streams and concatenate their tokens.

        Args:
            tactile_images: list of tensors, each ``(B, 3, H, W)``.
            tactile_masks: optional list of bool tensors ``(B,)``; True means
                the corresponding stream is present for that sample.

        Returns:
            tokens: ``(B, sum_streams * num_tokens_per_stream, feature_dim)``.
            pad_mask: ``(B, sum_streams * num_tokens_per_stream)`` bool.
        """
        if len(tactile_images) == 0:
            raise ValueError("ViTacEncoder received an empty tactile_images list.")

        all_tokens = []
        all_masks = []
        for idx, img in enumerate(tactile_images):
            tokens = self.encode_one(img)  # (B, T, D)
            bsize, num_tokens, _ = tokens.shape
            if tactile_masks is not None:
                stream_mask = tactile_masks[idx].to(dtype=torch.bool, device=tokens.device)
                stream_mask = stream_mask[:, None].expand(bsize, num_tokens)
            else:
                stream_mask = torch.ones(bsize, num_tokens, dtype=torch.bool, device=tokens.device)
            all_tokens.append(tokens)
            all_masks.append(stream_mask)

        tokens = torch.cat(all_tokens, dim=1)
        pad_mask = torch.cat(all_masks, dim=1)
        return tokens, pad_mask


class ViTacEncoder(BaseViTacEncoder):
    """Default stub ViTacEncoder: small CNN + learned token queries.

    Architecture (kept intentionally lightweight as a developer stub):

    * 4-stage strided conv stem (3 -> 32 -> 64 -> 128 -> 256 channels) that
      reduces a 224x224 input to a 14x14 feature map.
    * A learned ``num_tokens_per_stream``-by-``feature_dim`` query tensor
      combined with the spatially-pooled feature map via a single linear
      projection. The output tokens live in the encoder's *native* feature
      space; alignment to the prefix or FiLM space is done by
      :class:`TactileAlignmentHead` downstream.

    This module is fully randomly-initialized; it is meant to be replaced or
    pre-trained later. The interface (output shape and semantics) is the
    contract that the rest of the policy depends on.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_tokens_per_stream: int = 8,
        input_size: tuple[int, int] = (224, 224),
        hidden_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__(feature_dim=feature_dim, num_tokens_per_stream=num_tokens_per_stream)
        self.input_size = input_size

        c1, c2, c3, c4 = hidden_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.SiLU(),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c4),
            nn.SiLU(),
        )

        # Project pooled features to feature_dim, then expand into N tokens
        # with a learned token embedding table (acts as a tiny perceiver-style
        # latent set).
        self.proj = nn.Linear(c4, feature_dim)
        self.token_queries = nn.Parameter(torch.randn(num_tokens_per_stream, feature_dim) * 0.02)
        self.norm = nn.LayerNorm(feature_dim)

    def encode_one(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")

        if x.shape[-2:] != tuple(self.input_size):
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        feat = self.stem(x)  # (B, c4, h', w')
        # Spatial average pool to a single feature vector per stream.
        pooled = feat.mean(dim=(-2, -1))  # (B, c4)
        pooled = self.proj(pooled)  # (B, feature_dim)

        # Broadcast pooled feature against the learned token queries so that
        # different tokens specialize while still depending on the image.
        tokens = self.token_queries[None, :, :] + pooled[:, None, :]
        tokens = self.norm(tokens)
        return tokens


class TactileAlignmentHead(nn.Module):
    """MLP that lifts ViTacEncoder features into the SmolVLA prefix space.

    The CLAUDE.md requirement is to insert an MLP **after** the tactile
    encoder so the raw tactile features are projected into the same
    embedding space used by the cross-attention prefix (or the FiLM space).
    A single linear layer is insufficient: it cannot bridge the statistics
    of CNN feature maps and the language-model text embeddings. We use a
    two-layer GeLU MLP with a residual-friendly LayerNorm at the output.

    Shape: ``(B, T, in_dim)`` -> ``(B, T, out_dim)``.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(in_dim, out_dim) * hidden_mult
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return self.norm(h)


class FiLMHead(nn.Module):
    """Predicts FiLM gamma/beta from a pooled tactile feature.

    Produces multiplicative ``gamma`` and additive ``beta`` parameters of
    shape ``(B, target_dim)`` used to modulate the action-time embedding fed
    into the action expert::

        out = (1 + gamma) * x + beta

    The final linear layers are zero-initialized so that the modulation
    starts as an identity transform; the model gradually learns to leverage
    tactile signal during training without destabilising the warm-started
    SmolVLA expert.
    """

    def __init__(self, in_dim: int, target_dim: int, hidden_mult: int = 2) -> None:
        super().__init__()
        hidden = in_dim * hidden_mult
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        )
        self.to_gamma = nn.Linear(hidden, target_dim)
        self.to_beta = nn.Linear(hidden, target_dim)
        # Identity-initialization: gamma=0 (so (1+gamma)=1), beta=0.
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(pooled)
        gamma = self.to_gamma(h)
        beta = self.to_beta(h)
        return gamma, beta


class UniVTACEncoder(BaseViTacEncoder):
    """ViTacEncoder backed by a UniVTAC ResNet tactile backbone.

    Mirrors :class:`UniVTAC.encoder.network.Tactile`: a torchvision
    ``resnet18 / resnet34 / resnet50`` classifier-head is repurposed so that
    ``num_classes`` equals ``latent_dims``. The final fully-connected layer
    therefore yields a single ``(B, latent_dims)`` feature vector per image.

    Because all three SmolVLA fusion strategies (token / cross_attn / film)
    expect a token *sequence* ``(B, T, feature_dim)``, this encoder lifts the
    single feature vector into ``num_tokens_per_stream`` learned tokens via a
    perceiver-style additive query table (same idea as :class:`ViTacEncoder`).
    The downstream :class:`TactileAlignmentHead` then projects these tokens
    into either the SmolVLA prefix space (cross-attn) or the action expert
    hidden space (FiLM); for token-fusion the tokens are inserted directly
    into the prefix alongside SigLIP image embeddings (after going through
    the alignment head, since SigLIP outputs live at ``text_config.hidden_size``).

    Optional helpers:

    * :meth:`load_univtac_weights` -- load a checkpoint produced by the
      UniVTAC pre-training script. The checkpoint may store either the full
      ``Tactile`` model state dict (``backbone.*`` + ``decoders.*``) or only
      the backbone; we strip the ``backbone.`` prefix and load the matching
      keys with ``strict=False``.
    * ``freeze_backbone=True`` -- freezes all torchvision parameters so only
      the learned token queries and projection layer are trained.
    """

    _SUPPORTED_BACKBONES = ("resnet18", "resnet34", "resnet50")

    def __init__(
        self,
        feature_dim: int = 256,
        num_tokens_per_stream: int = 8,
        input_size: tuple[int, int] = (224, 224),
        backbone: str = "resnet18",
        latent_dims: int = 512,
        pretrained_path: str | Path | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(feature_dim=feature_dim, num_tokens_per_stream=num_tokens_per_stream)
        if backbone not in self._SUPPORTED_BACKBONES:
            raise ValueError(
                f"UniVTACEncoder backbone must be one of {self._SUPPORTED_BACKBONES}, got {backbone!r}"
            )

        # Lazy import so torchvision is only required when this encoder is used.
        from torchvision import models  # noqa: WPS433 (local import is intentional)

        self.backbone_name = backbone
        self.latent_dims = latent_dims
        self.input_size = input_size

        if backbone == "resnet18":
            self.backbone = models.resnet18(num_classes=latent_dims)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(num_classes=latent_dims)
        else:  # resnet50
            self.backbone = models.resnet50(num_classes=latent_dims)

        # latent -> feature_dim projection and N learned token queries.
        self.proj = nn.Linear(latent_dims, feature_dim)
        self.token_queries = nn.Parameter(
            torch.randn(num_tokens_per_stream, feature_dim) * 0.02
        )
        self.norm = nn.LayerNorm(feature_dim)

        if pretrained_path is not None:
            self.load_univtac_weights(pretrained_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def load_univtac_weights(self, ckpt_path: str | Path) -> None:
        """Load a UniVTAC pretrained checkpoint into the ResNet backbone.

        Accepts checkpoints that:
          * store the full ``Tactile`` state dict (keys prefixed by ``backbone.``
            or ``decoders.``) -- we keep only the backbone keys; or
          * store the bare ResNet state dict (no prefix).
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"UniVTAC checkpoint not found: {ckpt_path}")

        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(
                f"Unexpected checkpoint format at {ckpt_path}: expected a dict state_dict."
            )

        # Strip ``backbone.`` prefix if present.
        backbone_state: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                backbone_state[k[len("backbone."):]] = v
            elif k.startswith("decoders."):
                continue
            else:
                backbone_state[k] = v

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        log.info(
            "Loaded UniVTAC weights from %s (missing=%d, unexpected=%d)",
            ckpt_path, len(missing), len(unexpected),
        )
        if missing:
            log.debug("UniVTAC missing keys (first 5): %s", list(missing)[:5])
        if unexpected:
            log.debug("UniVTAC unexpected keys (first 5): %s", list(unexpected)[:5])

    def encode_one(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")

        if x.shape[-2:] != tuple(self.input_size):
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        # ResNet expects ImageNet-style 3-channel input; ViTac inputs live in
        # [-1, 1] (consistent with SigLIP normalization upstream). The backbone
        # accepts that range and the receptive field is the same.
        latent = self.backbone(x)  # (B, latent_dims)
        feat = self.proj(latent)   # (B, feature_dim)

        # Broadcast pooled feature against learned token queries so each token
        # specializes while still being conditioned on the image content.
        tokens = self.token_queries[None, :, :] + feat[:, None, :]
        tokens = self.norm(tokens)
        return tokens


def build_vitac_encoder(
    name: str,
    feature_dim: int = 256,
    num_tokens_per_stream: int = 8,
    **kwargs,
) -> BaseViTacEncoder:
    """Factory for ViTacEncoder variants.

    Supported names:

    * ``"stub" / "default" / "vitac"`` -- the lightweight :class:`ViTacEncoder`.
    * ``"univtac"`` -- :class:`UniVTACEncoder` (ResNet18/34/50 + token queries),
      optionally loaded from a UniVTAC pretrained checkpoint.
    """
    name = (name or "stub").lower()
    if name in {"stub", "default", "vitac"}:
        # Strip UniVTAC-only kwargs so the stub encoder doesn't choke on them.
        for k in ("backbone", "latent_dims", "pretrained_path", "freeze_backbone"):
            kwargs.pop(k, None)
        return ViTacEncoder(
            feature_dim=feature_dim,
            num_tokens_per_stream=num_tokens_per_stream,
            **kwargs,
        )
    if name in {"univtac", "univtac_resnet"}:
        return UniVTACEncoder(
            feature_dim=feature_dim,
            num_tokens_per_stream=num_tokens_per_stream,
            **kwargs,
        )
    raise ValueError(f"Unknown ViTacEncoder variant: {name!r}")
