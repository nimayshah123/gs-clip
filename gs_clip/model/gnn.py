"""
Graph Neural Network modules for encoding scene-graph structure.

Architecture used in GS-CLIP (from the paper):
  - 3-layer Graph Attention Network (GAT)
  - 8 attention heads per layer
  - Hidden / output dimension: 512 (matches CLIP ViT-B/32 embedding dim)
  - Residual connections + LayerNorm after each layer
  - GELU activations throughout

Node features are initialised from CLIP's token embedding table so the GNN
starts in a semantically meaningful space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------


class GraphConvLayer(nn.Module):
    """
    Standard graph convolution (GCN-style) with degree normalisation.

    Message passing:  h_i' = σ( (1/deg_i) Σ_{j∈N(i)} w_ij * W * h_j + b )
    """

    def __init__(self, in_dim: int, out_dim: int, use_edge_weights: bool = True):
        super().__init__()
        self.use_edge_weights = use_edge_weights
        self.linear = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            Node features  [num_nodes, in_dim]
            edge_index:   Edge list      [2, num_edges]
            edge_weight:  Edge weights   [num_edges]  (optional)
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        x_t = self.linear(x)                           # [N, out_dim]
        row, col = edge_index                           # sources, targets

        out = torch.zeros(x.size(0), x_t.size(1), device=x.device)

        messages = x_t[col]
        if edge_weight is not None and self.use_edge_weights:
            messages = messages * edge_weight.unsqueeze(-1)

        out.index_add_(0, row, messages)
        out = out + self.bias

        # Normalise by out-degree to stabilise training
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=x.device))
        out = out / deg.unsqueeze(-1).clamp(min=1.0)

        return out


class GraphAttentionLayer(nn.Module):
    """
    Multi-head Graph Attention (GAT) layer.

    Attention coefficients:
        e_ij = LeakyReLU( a^T [W h_i || W h_j] )
        α_ij = softmax_j( e_ij )

    Output:  h_i' = concat_{k=1}^{K} σ( Σ_j α_ij^k W^k h_j )
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_weights: bool = True,
    ):
        """
        Args:
            in_dim:           Input node feature dimension.
            out_dim:          Output node feature dimension (must divide evenly
                              by num_heads so each head gets out_dim // num_heads).
            num_heads:        Number of attention heads.
            dropout:          Dropout on attention weights.
            use_edge_weights: Scale attention scores by edge weights when provided.
        """
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_edge_weights = use_edge_weights

        # Shared linear projection for all heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention vector per head: [1, K, 2*head_dim]
        self.attn = nn.Parameter(torch.empty(1, num_heads, 2 * self.head_dim))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = x.size(0)
        row, col = edge_index

        # Project and reshape: [N, K, head_dim]
        h = self.W(x).view(N, self.num_heads, self.head_dim)

        # Concatenate source and target per edge: [E, K, 2*head_dim]
        h_cat = torch.cat([h[row], h[col]], dim=-1)

        # Compute attention scores: [E, K]
        alpha = (h_cat * self.attn).sum(dim=-1)
        alpha = self.leaky_relu(alpha)

        if edge_weight is not None and self.use_edge_weights:
            alpha = alpha * edge_weight.unsqueeze(-1)

        # Softmax normalised per destination node (using scatter softmax)
        # Simple per-graph softmax — sufficient for small graphs
        alpha_soft = F.softmax(alpha, dim=0)
        alpha_soft = self.dropout(alpha_soft)

        # Aggregate neighbour features
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        for i in range(edge_index.size(1)):
            r, c = row[i], col[i]
            out[r] += alpha_soft[i].unsqueeze(-1) * h[c]

        return out.view(N, self.num_heads * self.head_dim)


# ---------------------------------------------------------------------------
# Full GNN encoder
# ---------------------------------------------------------------------------


class CompositionGNN(nn.Module):
    """
    Stacked GNN that encodes a dependency-parsed scene graph into node
    embeddings that capture predicate–argument structure.

    Used in GS-CLIP Stage I (Visual Genome) and Stage II (COCO).

    Architecture:
        input_proj  → [gnn_layer + residual + LayerNorm] × num_layers → output_proj
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 3,
        gnn_type: str = "gat",
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim:  Node feature dimension coming in (equals CLIP embed_dim).
            hidden_dim: Internal dimension of each GNN layer.
            output_dim: Output dimension (must equal CLIP embed_dim for fusion).
            num_layers: Number of GNN message-passing steps (paper uses 3).
            gnn_type:   "gat" (Graph Attention) or "gcn" (plain GCN).
            num_heads:  Attention heads for GAT (paper uses 8).
            dropout:    Dropout rate applied after each layer.
        """
        super().__init__()
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            if gnn_type == "gat":
                layer = GraphAttentionLayer(
                    hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout
                )
            elif gnn_type == "gcn":
                layer = GraphConvLayer(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"gnn_type must be 'gat' or 'gcn', got '{gnn_type}'")
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            Node features   [num_nodes, input_dim]
            edge_index:   Edges           [2, num_edges]
            edge_weight:  Edge weights    [num_edges]   (optional)
        Returns:
            Node embeddings  [num_nodes, output_dim]
        """
        h = self.activation(self.input_proj(x))

        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            h_new = self.dropout(gnn_layer(h, edge_index, edge_weight))
            # Residual connection when dimensions match
            if h.shape == h_new.shape:
                h_new = h + h_new
            h = layer_norm(self.activation(h_new))

        return self.output_proj(h)

    def graph_embedding(
        self, node_embeddings: torch.Tensor, pooling: str = "mean"
    ) -> torch.Tensor:
        """Pool node embeddings into a single graph-level vector."""
        if pooling == "mean":
            return node_embeddings.mean(dim=0)
        elif pooling == "max":
            return node_embeddings.max(dim=0)[0]
        elif pooling == "sum":
            return node_embeddings.sum(dim=0)
        raise ValueError(f"Unknown pooling '{pooling}'")
