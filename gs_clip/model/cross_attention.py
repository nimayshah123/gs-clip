"""
Cross-attention fusion and targeted attention-head surgery for GS-CLIP.

Two complementary approaches to inject scene-graph structure into CLIP:

1. CrossAttentionFusion (used in GS-CLIP main model)
   GNN node embeddings become K and V; CLIP token states become Q.
   Residual addition fuses structural information without disrupting
   the pretrained token representations.

       H_fused = H_CLIP + MultiHead(Q=H_CLIP, K=H_GNN, V=H_GNN)

2. TargetHeadAttn (alternative: surgical single-head training)
   Replaces only the Q_h and K_h projections of one attention head
   in one transformer layer with trainable parameters, keeping
   everything else frozen. Training signal: KL(adjacency || attn_weights).
   Extremely parameter-efficient: ~65K trainable params.
"""

import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Cross-attention fusion (main GS-CLIP architecture)
# ---------------------------------------------------------------------------


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention that fuses GNN node embeddings into CLIP tokens.

    Queries  = CLIP hidden states (token-level)
    Keys/Vals = GNN node embeddings (graph-level, one per scene-graph node)

    A residual connection and LayerNorm preserve the pretrained CLIP features
    while allowing graph structure to modulate token representations.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            dim:       Token/node embedding dimension (must equal CLIP embed_dim).
            num_heads: Number of attention heads.
            dropout:   Attention dropout.
        """
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        clip_tokens: torch.Tensor,
        gnn_nodes: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            clip_tokens: CLIP token hidden states  [B, T, dim]
            gnn_nodes:   GNN node embeddings       [B, N, dim]
            node_mask:   Boolean mask, True = valid node  [B, N]  (optional)
        Returns:
            Fused token representations            [B, T, dim]
        """
        B = clip_tokens.size(0)

        # Project to Q (from CLIP), K/V (from GNN)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(clip_tokens))  # [B, H, T, d_h]
        K = split_heads(self.k_proj(gnn_nodes))    # [B, H, N, d_h]
        V = split_heads(self.v_proj(gnn_nodes))    # [B, H, N, d_h]

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, T, N]

        if node_mask is not None:
            # Mask padding nodes by setting their scores to -inf
            mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                         # [B, H, T, d_h]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
        out = self.out_proj(out)

        # Residual + LayerNorm
        return self.layer_norm(clip_tokens + out)


# ---------------------------------------------------------------------------
# 2. Surgical single-head training (alternative / ablation)
# ---------------------------------------------------------------------------


class TargetHeadAttn(nn.Module):
    """
    Wraps an existing MultiheadAttention module and makes only the Q and K
    projections of one specified head trainable.

    All other parameters (V, output projection, remaining heads) remain frozen.
    This gives ~2 × head_dim × embed_dim ≈ 65K trainable parameters — roughly
    300× fewer than a full LoRA setup.

    Usage:
        wrapper = TargetHeadAttn(block.attn, head=0, head_dim=64, embed_dim=512)
        block.attn = wrapper  # or patch block.forward as shown below
    """

    def __init__(
        self,
        original_attn: nn.Module,
        head: int,
        head_dim: int,
        embed_dim: int,
    ):
        """
        Args:
            original_attn: The MultiheadAttention to wrap (from open_clip).
            head:          Index of the head to make trainable (0-indexed).
            head_dim:      Dimension per head (embed_dim // num_heads).
            embed_dim:     Full embedding dimension.
        """
        super().__init__()
        self.original = original_attn
        self.head = head
        self.head_dim = head_dim
        self.embed_dim = embed_dim

        # Extract and register Q_h, K_h as learnable parameters.
        # in_proj_weight layout: [Q_all | K_all | V_all], each block [D, D].
        # Head h occupies rows [h*d_h : (h+1)*d_h] within each block.
        with torch.no_grad():
            W = original_attn.in_proj_weight  # [3D, D]
            D, d_h = embed_dim, head_dim
            self.Q_h = nn.Parameter(W[head * d_h: (head + 1) * d_h].clone())
            self.K_h = nn.Parameter(W[D + head * d_h: D + (head + 1) * d_h].clone())

        # Keep a frozen snapshot of the full weight for the other heads/V
        self.register_buffer("W_frozen", W.clone())

    def _full_in_proj_weight(self) -> torch.Tensor:
        """Reconstruct in_proj_weight with trainable Q_h, K_h spliced in."""
        W = self.W_frozen.clone()
        D, d_h, h = self.embed_dim, self.head_dim, self.head
        W[h * d_h: (h + 1) * d_h] = self.Q_h
        W[D + h * d_h: D + (h + 1) * d_h] = self.K_h
        return W

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        orig_weight = self.original.in_proj_weight
        self.original.in_proj_weight = nn.Parameter(
            self._full_in_proj_weight(), requires_grad=True
        )
        out, weights = F.multi_head_attention_forward(
            x, x, x,
            self.embed_dim,
            self.original.num_heads,
            self.original.in_proj_weight,
            self.original.in_proj_bias,
            None, None, False, 0.0,
            self.original.out_proj.weight,
            self.original.out_proj.bias,
            training=self.training,
            need_weights=True,
            average_attn_weights=False,  # keep per-head: [B, heads, L, L]
        )
        self.original.in_proj_weight = orig_weight
        return out, weights


def patch_resblock_for_attn_capture(
    block: nn.Module,
    head_attn_wrapper: TargetHeadAttn,
    attn_buffer: list,
) -> None:
    """
    Monkey-patch a CLIP ResidualAttentionBlock so that:
      - The forward method calls head_attn_wrapper instead of the original MHA.
      - Attention weights are stored in attn_buffer[0] for loss computation.

    Args:
        block:             CLIP ResidualAttentionBlock to patch.
        head_attn_wrapper: TargetHeadAttn instance that replaces self-attention.
        attn_buffer:       A single-element list; attn_buffer[0] gets the weights.
    """

    def patched_forward(self, x, attn_mask=None):
        residual = x
        x_norm = self.ln_1(x)

        # open_clip uses seq-first layout [L, B, D]
        attn_out, attn_weights = head_attn_wrapper(
            x_norm.transpose(0, 1) if x_norm.dim() == 3 else x_norm,
            attn_mask=attn_mask,
            need_weights=True,
        )
        if attn_out.dim() == 3:
            attn_out = attn_out.transpose(0, 1)

        attn_buffer[0] = attn_weights  # [B, num_heads, L, L]
        x = residual + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

    block.forward = types.MethodType(patched_forward, block)
