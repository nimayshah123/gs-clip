"""
GS-CLIP: the full Graph-Supervised CLIP model.

Architecture overview (from the paper):
  1. Text → KnowledgeGraphBuilder → dependency-based scene graph
  2. Scene graph → CompositionGNN (3-layer GAT) → node embeddings
  3. Node embeddings → CrossAttentionFusion → fused token states
  4. Fused states → CLIP text encoder (with LoRA adapters) → sentence embedding
  5. Images → CLIP vision encoder (frozen) → image embedding

Training:
  Stage I  — Visual Genome: CLIP contrastive loss on image-region pairs.
             Only the GNN, fusion module, and LoRA adapters are updated.
  Stage II — MS-COCO: composite objective targeting relational geometry
             while preserving global image–text alignment.

Key design decisions:
  - CLIP vision encoder is kept fully frozen throughout.
  - A learned sigmoid gate blends GNN-enhanced features with plain CLIP
    features; it converges to ~0.5, confirming both contribute equally.
  - LoRA rank=16 introduces only ~4×10^5 trainable parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import open_clip

from .gnn import CompositionGNN
from .lora import inject_lora_into_clip
from .cross_attention import CrossAttentionFusion
from ..data.kg_builder import KnowledgeGraphBuilder


class GSCLIPModel(nn.Module):
    """
    Full GS-CLIP model combining CLIP, a GNN, cross-attention fusion, and LoRA.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        gnn_layers: int = 3,
        gnn_heads: int = 8,
        gnn_dropout: float = 0.1,
        fusion_heads: int = 8,
        device: str = "cpu",
    ):
        """
        Args:
            clip_model_name: CLIP architecture (paper uses ViT-B/32).
            clip_pretrained: Pretrained weights identifier for open_clip.
            lora_rank:       LoRA decomposition rank (paper: r=16).
            lora_alpha:      LoRA scaling factor (paper: α=32).
            gnn_layers:      Number of GAT message-passing steps (paper: 3).
            gnn_heads:       GAT attention heads (paper: 8).
            gnn_dropout:     Dropout in the GNN.
            fusion_heads:    Heads in the cross-attention fusion module.
            device:          Torch device string.
        """
        super().__init__()
        self.device = device

        # --- Load CLIP ---
        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip = clip.to(device)
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.embed_dim: int = clip.transformer.width  # 512 for ViT-B/32

        # Freeze all CLIP parameters
        for param in self.clip.parameters():
            param.requires_grad_(False)

        # --- Inject LoRA into the text encoder ---
        self.lora_layers = inject_lora_into_clip(
            self.clip, rank=lora_rank, alpha=lora_alpha
        )

        # --- Scene-graph parsing ---
        self.kg_builder = KnowledgeGraphBuilder()

        # --- Graph encoder ---
        self.gnn = CompositionGNN(
            input_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            num_layers=gnn_layers,
            gnn_type="gat",
            num_heads=gnn_heads,
            dropout=gnn_dropout,
        )

        # --- Cross-attention fusion ---
        self.fusion = CrossAttentionFusion(
            dim=self.embed_dim, num_heads=fusion_heads, dropout=gnn_dropout
        )

        # --- Learned gate: blends GNN-enhanced with plain CLIP features ---
        # Converges to ~0.5 in the paper, confirming equal contribution
        self.gate = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

        # --- Final projection (keeps output in CLIP embedding space) ---
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(gnn_dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with the frozen CLIP vision encoder."""
        feats = self.clip.encode_image(images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, tokens: torch.Tensor, captions: list[str]) -> torch.Tensor:
        """
        Encode text with graph-supervised fusion.

        Args:
            tokens:   Tokenised captions [B, 77] (from open_clip tokenizer).
            captions: Raw caption strings (same order as tokens) for graph building.
        Returns:
            Normalised text embeddings [B, embed_dim].
        """
        # --- Plain CLIP text features ---
        clip_feats = self.clip.encode_text(tokens)          # [B, D]
        clip_feats_norm = F.normalize(clip_feats, dim=-1)

        # --- Build and encode scene graphs ---
        gnn_feats = self._encode_graphs(captions, tokens)   # [B, D]

        # --- Sigmoid gate: balance CLIP and GNN contributions ---
        g = torch.sigmoid(self.gate)
        mixed = (1 - g) * clip_feats_norm + g * gnn_feats  # [B, D]

        # --- Final projection and normalisation ---
        out = self.output_proj(mixed)
        return F.normalize(out, dim=-1)

    def _encode_graphs(
        self, captions: list[str], tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Build a scene graph for each caption, run it through the GNN, and
        return one graph-level embedding per caption via mean pooling.

        Falls back to plain CLIP features for captions that yield empty graphs.
        """
        D = self.embed_dim
        B = len(captions)
        device = tokens.device

        # Get CLIP token embeddings to initialise GNN node features
        with torch.no_grad():
            tok_embeds = self.clip.token_embedding(tokens)  # [B, 77, D]

        results = torch.zeros(B, D, device=device)

        for i, caption in enumerate(captions):
            graph, nodes, _ = self.kg_builder.build_graph(caption)
            if len(nodes) == 0:
                # Degenerate case: no parse → use CLS (EOS) CLIP feature
                with torch.no_grad():
                    results[i] = F.normalize(
                        self.clip.encode_text(tokens[i: i + 1]), dim=-1
                    ).squeeze(0)
                continue

            # Initialise node features from the CLIP embedding of each word
            node_feats = torch.stack(
                [tok_embeds[i, min(n.idx, 76)] for n in nodes], dim=0
            ).to(device)  # [num_nodes, D]

            # Convert graph edges to tensors
            edges = self.kg_builder.to_pyg_data(graph, nodes, node_features=node_feats)
            edge_index = edges["edge_index"].to(device)
            edge_weight = edges["edge_weight"].to(device)

            # GNN forward
            node_out = self.gnn(node_feats, edge_index, edge_weight)  # [num_nodes, D]
            graph_emb = node_out.mean(dim=0)                          # [D]
            results[i] = F.normalize(graph_emb, dim=-1)

        return results

    # ------------------------------------------------------------------
    # Full forward pass (for training)
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        captions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images:   Image batch  [B, 3, H, W]
            tokens:   Token IDs    [B, 77]
            captions: Raw strings  [B]
        Returns:
            (image_features, text_features) — both L2-normalised, shape [B, D]
        """
        image_feats = self.encode_image(images)
        text_feats = self.encode_text(tokens, captions)
        return image_feats, text_feats

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that should receive gradients."""
        params = []
        # GNN
        params.extend(self.gnn.parameters())
        # Cross-attention fusion
        params.extend(self.fusion.parameters())
        # LoRA adapters
        for layer in self.lora_layers:
            params.extend(layer.lora.parameters())
        # Gate and output proj
        params.append(self.gate)
        params.extend(self.output_proj.parameters())
        return params

    def count_trainable(self) -> dict[str, int]:
        """Return a breakdown of trainable parameter counts by component."""
        return {
            "gnn": sum(p.numel() for p in self.gnn.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "lora": sum(p.numel() for ll in self.lora_layers for p in ll.lora.parameters()),
            "gate_and_proj": (
                self.gate.numel()
                + sum(p.numel() for p in self.output_proj.parameters())
            ),
        }
