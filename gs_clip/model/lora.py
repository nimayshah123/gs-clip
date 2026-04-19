"""
LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of CLIP.

Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., ICLR 2022)

Instead of fine-tuning the full weight matrix W ∈ R^{d_out × d_in}, we learn:
    ΔW = B @ A,  where B ∈ R^{d_out × r}, A ∈ R^{r × d_in}, r << d_in

Forward:  y = W @ x + (α/r) * B @ A @ x

In GS-CLIP: LoRA adapters are inserted into fc1 and fc2 projections of all 12
transformer blocks in CLIP's text encoder, introducing ~4×10^5 trainable
parameters (0.3% of the CLIP backbone).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Single low-rank adaptation: computes (α/r) * B @ A @ x.

    Initialized so that ΔW = 0 at the start of training: A uses Kaiming
    uniform init, B is zeroed. This ensures the adapted model starts
    identical to the frozen pretrained model.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features:  Input dimension (d_in).
            out_features: Output dimension (d_out).
            rank:         Decomposition rank r. Lower → fewer parameters.
            alpha:        Scaling factor. Effective scale = alpha / rank.
            dropout:      Optional dropout on the LoRA input path.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # A: [r, d_in] — initialized with Kaiming uniform
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # B: [d_out, r] — initialized to zero so ΔW starts at zero
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_features]
        Returns:
            LoRA update [..., out_features], scaled by alpha/rank.
        """
        x = self.dropout(x)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a parallel LoRA path.

        y = W @ x + b + LoRA(x)

    The base layer (W, b) is frozen; only the LoRA matrices are trained.
    Exposes weight and bias attributes directly so that code that accesses
    layer.weight (e.g., open_clip internals) continues to work correctly.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        train_base: bool = False,
    ):
        """
        Args:
            linear:     Pre-trained nn.Linear to wrap.
            rank:       LoRA rank.
            alpha:      LoRA scaling factor.
            dropout:    Dropout on the LoRA path.
            train_base: If True, also fine-tune the base weights (expensive).
        """
        super().__init__()

        # Keep a reference to the original layer so its forward logic is reused
        self._base = linear
        self._base.weight.requires_grad_(train_base)
        if self._base.bias is not None:
            self._base.bias.requires_grad_(train_base)

        # Expose scalar attributes expected by open_clip
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight  # frozen view
        self.bias = linear.bias if linear.bias is not None else None

        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._base(x) + self.lora(x)

    def merge_weights(self) -> None:
        """Absorb LoRA into the base weight in-place (for inference speed)."""
        with torch.no_grad():
            delta = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            self._base.weight.data += delta


def inject_lora_into_clip(
    clip_model: nn.Module, rank: int = 16, alpha: float = 32.0
) -> list[LoRALinear]:
    """
    Replace the fc1 and fc2 MLP projections in every transformer block of
    CLIP's text encoder with LoRA-wrapped versions.

    Only the LoRA matrices are trainable; the original CLIP weights are frozen.

    Args:
        clip_model: A loaded open_clip model.
        rank:       LoRA rank (paper uses r=16).
        alpha:      LoRA scaling (paper uses α=32).

    Returns:
        List of all LoRALinear layers that were injected (for easy param access).
    """
    lora_layers: list[LoRALinear] = []

    for block in clip_model.transformer.resblocks:
        block.mlp.c_fc = LoRALinear(block.mlp.c_fc, rank=rank, alpha=alpha)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, rank=rank, alpha=alpha)
        lora_layers.append(block.mlp.c_fc)
        lora_layers.append(block.mlp.c_proj)

    return lora_layers
