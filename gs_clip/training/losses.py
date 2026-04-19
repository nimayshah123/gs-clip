"""
Loss functions for GS-CLIP training.

Stage I (Visual Genome) uses:
  - contrastive_loss     — standard CLIP InfoNCE
  - kl_graph_loss        — KL(adjacency || attention_weights)  [surgical variant]

Stage II (MS-COCO) uses a weighted composite (Eq. 9 in the paper):
  L = λ_s L_struct + λ_h L_hneg + λ_i L_iso + λ_r L_rel + λ_d L_single

  L_struct : WL-fingerprint reweighted contrastive loss
  L_hneg   : CLIC-style hard-negative contrastive loss
  L_iso    : Graph-similarity weighted isotropy (same-image caption pulling)
  L_rel    : Token-pair relation preservation (hinge on edge-connected tokens)
  L_single : Standard InfoNCE on single images (prevents catastrophic forgetting)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage I losses
# ---------------------------------------------------------------------------


def contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP contrastive) loss.

    Both inputs must be L2-normalised embeddings of shape [B, D].
    The diagonal of the similarity matrix is the set of positive pairs.

    Args:
        image_features: [B, D] normalised image embeddings.
        text_features:  [B, D] normalised text embeddings.
        temperature:    Logit scaling; paper trains with learned logit_scale.
    Returns:
        Scalar loss (mean of image-to-text and text-to-image cross-entropies).
    """
    logits = (image_features @ text_features.T) / temperature  # [B, B]
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2.0


def kl_graph_loss(
    adj_batch: torch.Tensor,
    edge_mask: torch.Tensor,
    token_ids: torch.Tensor,
    attn_weights: torch.Tensor,
    target_head: int = 0,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    KL divergence between scene-graph adjacency and a single attention head's
    attention distribution. Used in the surgical TargetHeadAttn training variant.

    KL(P_graph || Q_attn): zero-forcing (forward KL) — forces the attention
    head to put mass on every graph edge, not just the dominant one.

    Args:
        adj_batch:    Binary adjacency matrices  [B, L, L].
        edge_mask:    1 for token rows that have at least one graph edge [B, L].
        token_ids:    BPE token IDs (for masking BOS/EOS/PAD) [B, L].
        attn_weights: Per-head attention weights [B, num_heads, L, L].
        target_head:  Which head's distribution to align with the graph.
        temperature:  Softens peaked attention before computing KL.
    Returns:
        Scalar KL loss averaged over active (edge-bearing, content) positions.
    """
    BOS, EOS, PAD = 49406, 49407, 0

    # Extract and temperature-soften the target head's attention
    Q_raw = attn_weights[:, target_head, :, :]                 # [B, L, L]
    log_Q = torch.log(Q_raw.clamp(min=1e-9))
    Q = F.softmax(log_Q / temperature, dim=-1)                 # [B, L, L]

    # Row-normalise adjacency to get target distribution P
    row_sums = adj_batch.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    P = adj_batch / row_sums                                    # [B, L, L]

    # Mask out BOS / EOS / PAD positions (no gradient from them)
    content_mask = (
        (token_ids != BOS) & (token_ids != EOS) & (token_ids != PAD)
    ).float()                                                   # [B, L]

    active = edge_mask * content_mask                           # [B, L]

    # KL(P || Q) per position: Σ_j P[j] * (log P[j] - log Q[j])
    kl = (P * (torch.log(P.clamp(min=1e-9)) - torch.log(Q.clamp(min=1e-9)))).sum(-1)
    kl_masked = kl * active                                     # [B, L]

    n_active = active.sum().clamp(min=1.0)
    return kl_masked.sum() / n_active


# ---------------------------------------------------------------------------
# Stage II losses
# ---------------------------------------------------------------------------


def structural_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    wl_weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    L_struct: Weisfeiler–Lehman graph fingerprint reweighted contrastive loss.

    Structurally distinct captions get higher contrastive penalty in the
    denominator, increasing separation between role-swapped pairs.

    wl_weights[i, j] encodes how structurally different caption j is from
    caption i (higher = more different). A value of 1.0 is neutral (same as
    vanilla contrastive); values > 1 penalise structurally similar distractors
    less and structurally different ones more.

    Args:
        image_features: [B, D] normalised.
        text_features:  [B, D] normalised.
        wl_weights:     [B, B] structural similarity reweighting matrix.
        temperature:    Logit scaling.
    Returns:
        Scalar loss.
    """
    sim = (image_features @ text_features.T) / temperature     # [B, B]
    targets = torch.arange(sim.size(0), device=sim.device)

    # Add structural penalty to off-diagonal logits
    sim_adj = sim + torch.log(1.0 + wl_weights.clamp(min=0.0))

    loss_i2t = F.cross_entropy(sim_adj, targets)
    loss_t2i = F.cross_entropy(sim_adj.T, targets)
    return (loss_i2t + loss_t2i) / 2.0


def hard_negative_loss(
    pos_sims: torch.Tensor,
    neg_sims: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    L_hneg: CLIC-style hard-negative contrastive loss.

    Hard negatives are generated offline by concatenating two unrelated images
    and swapping a noun between their captions (see the CLIC paper).

    pos_sims[i] = similarity(image_i, positive_caption_i)
    neg_sims[i] = similarity(image_i, hard_negative_caption_i)

    Args:
        pos_sims: [B] positive pair cosine similarities.
        neg_sims: [B] hard-negative pair cosine similarities.
        temperature: Logit scaling.
    Returns:
        Scalar cross-entropy (positive should outscore the hard negative).
    """
    logits = torch.stack([pos_sims, neg_sims], dim=1) / temperature  # [B, 2]
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


def isotropy_loss(
    text_features_a: torch.Tensor,
    text_features_b: torch.Tensor,
    graph_similarity: torch.Tensor,
) -> torch.Tensor:
    """
    L_iso: graph-similarity weighted caption pulling for the same image.

    Captions describing the same image are attracted to each other
    proportionally to how similar their scene graphs are. This improves
    embedding isotropy (uniform coverage of the embedding sphere).

    Args:
        text_features_a: [B, D] embeddings for one caption per image.
        text_features_b: [B, D] embeddings for a second caption of the same image.
        graph_similarity: [B] WL-based similarity weight ∈ [0, 1].
    Returns:
        Scalar loss (weighted mean of cosine distances).
    """
    cos_sim = (text_features_a * text_features_b).sum(dim=-1)  # [B]
    cos_dist = 1.0 - cos_sim
    return (graph_similarity * cos_dist).mean()


def relational_loss(
    token_hidden: torch.Tensor,
    edge_pairs: list[tuple[int, int]],
    margin: float = 0.5,
) -> torch.Tensor:
    """
    L_rel: token-pair relation preservation.

    Token pairs that are connected in the scene graph should maintain
    positively-correlated hidden representations. A hinge loss is applied
    whenever the cosine similarity between a connected pair falls below margin.

    This directly penalises the embedding geometry causing relational collapse.

    Args:
        token_hidden: Hidden states of all tokens [T, D] (single example).
        edge_pairs:   List of (source_token_idx, target_token_idx) from scene graph.
        margin:       Minimum acceptable cosine similarity between connected tokens.
    Returns:
        Scalar hinge loss averaged over edges.
    """
    if len(edge_pairs) == 0:
        return token_hidden.new_tensor(0.0)

    losses = []
    for i, j in edge_pairs:
        if i >= token_hidden.size(0) or j >= token_hidden.size(0):
            continue
        cos_sim = F.cosine_similarity(
            token_hidden[i].unsqueeze(0), token_hidden[j].unsqueeze(0)
        )
        losses.append(F.relu(margin - cos_sim))

    if not losses:
        return token_hidden.new_tensor(0.0)
    return torch.stack(losses).mean()


def composite_stage2_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    pos_sims: torch.Tensor,
    neg_sims: torch.Tensor,
    text_features_alt: torch.Tensor,
    graph_similarity: torch.Tensor,
    token_hidden: torch.Tensor,
    edge_pairs: list[tuple[int, int]],
    wl_weights: torch.Tensor,
    lambdas: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Full Stage II composite loss (Eq. 9 in the paper):

        L = λ_s L_struct + λ_h L_hneg + λ_i L_iso + λ_r L_rel + λ_d L_single

    Default λ values reproduce the paper's experimental setup.

    Returns:
        Dict with individual loss terms and the total weighted loss.
    """
    if lambdas is None:
        lambdas = {"struct": 1.0, "hneg": 1.0, "iso": 0.5, "rel": 0.5, "single": 1.0}

    l_struct = structural_contrastive_loss(image_features, text_features, wl_weights)
    l_hneg   = hard_negative_loss(pos_sims, neg_sims)
    l_iso    = isotropy_loss(text_features, text_features_alt, graph_similarity)
    l_rel    = relational_loss(token_hidden, edge_pairs)
    l_single = contrastive_loss(image_features, text_features)

    total = (
        lambdas["struct"] * l_struct
        + lambdas["hneg"]  * l_hneg
        + lambdas["iso"]   * l_iso
        + lambdas["rel"]   * l_rel
        + lambdas["single"]* l_single
    )

    return {
        "total":   total,
        "struct":  l_struct,
        "hneg":    l_hneg,
        "iso":     l_iso,
        "rel":     l_rel,
        "single":  l_single,
    }
