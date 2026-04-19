# GS-CLIP: Graph-Supervised Adaptation of CLIP

**Graph-Supervised Adaptation of CLIP Improves Relational Compositionality by Reshaping Language Embedding Geometry**

*ECCV 2026 Submission — Paper #13326*

---

## Overview

Contrastive vision-language models like CLIP suffer from **relational embedding collapse**: sentence pairs that share the same predicate but differ only in argument role assignment (e.g., *"a dog chases a cat"* vs. *"a cat chases a dog"*) are assigned nearly identical embeddings. In CLIP ViT-B/32, the mean cosine similarity across such pairs is **0.971** — almost indistinguishable from the same sentence twice.

**GS-CLIP** addresses this by injecting relational computation directly into CLIP's text encoder:

1. Dependency-parsed **scene graphs** are built from captions.
2. A 3-layer **Graph Attention Network (GAT)** encodes the graph into structurally-aware node embeddings.
3. **Cross-attention fusion** injects GNN features into CLIP's token hidden states.
4. **LoRA adapters** (rank 16, ~0.3% of CLIP parameters) enable parameter-efficient fine-tuning.

Training proceeds in two stages: **Stage I** on Visual Genome (~108K images), then **Stage II** on MS-COCO with a composite geometry-aware objective.

### Key Results

| Metric | CLIP ViT-B/32 | GS-CLIP | Improvement |
|--------|:---:|:---:|:---:|
| Intra-Predicate Similarity (IPS) ↓ | 0.7822 | 0.379 | **−52%** |
| Random-pair similarity ↓ | 0.6844 | 0.2647 | **−61%** |
| ARO VG-Relation ↑ | 59.9% | 88.95% | **+29 pp** |
| Role-swap discrimination ↑ | 34.8% | 65.2% | **+30 pp** |
| Forced-choice (threshold 0.95) ↑ | 10.6% | 30.8% | **+20 pp** |

---

## Repository Structure

```
gs-clip/
├── gs_clip/                 # Main Python package
│   ├── model/
│   │   ├── gs_clip.py       # Full GS-CLIP model (entry point)
│   │   ├── gnn.py           # Graph Attention / Convolution layers
│   │   ├── lora.py          # LoRA adapters for parameter-efficient fine-tuning
│   │   └── cross_attention.py  # Cross-attention fusion + surgical head training
│   ├── data/
│   │   ├── kg_builder.py    # Scene graph construction via dependency parsing
│   │   └── datasets.py      # Winoground, SugarCrepe, Visual Genome loaders
│   ├── training/
│   │   └── losses.py        # All loss functions (Stage I and Stage II)
│   └── evaluation/
│       ├── evaluator.py     # Unified evaluator for all benchmarks
│       └── metrics.py       # IPS, role-swap accuracy, SVO win rate
├── scripts/
│   ├── train_stage1.py      # Stage I: Visual Genome training
│   ├── train_stage2.py      # Stage II: COCO geometry-aware fine-tuning
│   ├── evaluate.py          # Run all benchmarks
│   ├── download_data.py     # Download Winoground & SugarCrepe
│   └── compare_baselines.py # Compare results to published baselines
├── notebooks/               # Jupyter notebooks (exploration / analysis)
├── results/
│   ├── figures/             # Training curves and diagnostic plots
│   └── json/                # Benchmark result files
├── configs/
│   └── default.yaml         # All hyperparameters in one place
└── paper/                   # The ECCV submission PDF
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

PyTorch Geometric also needs a GPU-compatible install:
```bash
pip install torch-geometric
```

### 2. Download evaluation datasets

```bash
# SugarCrepe (free, no account needed)
python scripts/download_data.py --dataset sugarcrepe

# Winoground (requires HuggingFace token)
export HF_TOKEN=hf_your_token_here
python scripts/download_data.py --dataset winoground
```

Visual Genome (Stage I training) and MS-COCO (Stage II) must be downloaded manually — see `scripts/download_data.py` for instructions.

### 3. Evaluate the baseline

```bash
# Evaluate CLIP ViT-B/32 baseline on all benchmarks
python scripts/evaluate.py
```

### 4. Train GS-CLIP

```bash
# Stage I: Visual Genome pre-training (~5000 steps)
python scripts/train_stage1.py --vg_root data/visual_genome

# Stage II: COCO geometry-aware fine-tuning
python scripts/train_stage2.py --stage1_ckpt checkpoints/stage1/final.pt \
                                --coco_root data/coco

# Evaluate your trained model
python scripts/evaluate.py --checkpoint checkpoints/stage2/final.pt
```

### 5. Compare against published baselines

```bash
python scripts/compare_baselines.py results/json/eval_results.json
python scripts/compare_baselines.py results/json/eval_results.json --latex
```

---

## Architecture Details

### Scene Graph Construction (`gs_clip/data/kg_builder.py`)

Captions are parsed with spaCy's dependency parser. Nodes are tokens; edges are dependency relations weighted by compositional importance:

| Relation | Weight | Reason |
|----------|:---:|---|
| `nsubj`, `dobj` | 1.0 | Core predicate–argument links |
| `amod`, `compound` | 0.9 | Compositional modifiers |
| `prep`, `pobj` | 0.6–0.7 | Spatial/semantic relations |
| `det`, `aux` | 0.3–0.5 | Functional/structural |

### GNN Encoder (`gs_clip/model/gnn.py`)

3-layer Graph Attention Network:
- 8 attention heads per layer, hidden dim = 512
- Residual connections + LayerNorm after each layer
- GELU activations
- Node features initialised from CLIP's token embedding table

### LoRA Adapters (`gs_clip/model/lora.py`)

Inserted into `fc1` and `fc2` of all 12 transformer blocks:
- Rank r = 16, scaling α = 32
- ~4 × 10⁵ trainable parameters (0.3% of CLIP backbone)
- Initialised so ΔW = 0 at training start

### Cross-Attention Fusion (`gs_clip/model/cross_attention.py`)

```
H_fused = H_CLIP + MultiHead(Q=H_CLIP, K=H_GNN, V=H_GNN)
```

Queries come from CLIP token states; keys and values from GNN node embeddings. A residual connection and LayerNorm preserve pretrained CLIP features.

### Stage II Loss (Eq. 9 from paper)

```
L = λ_s L_struct + λ_h L_hneg + λ_i L_iso + λ_r L_rel + λ_d L_single
```

| Term | Description |
|------|-------------|
| `L_struct` | WL-graph-fingerprint reweighted contrastive |
| `L_hneg` | CLIC hard-negative contrastive |
| `L_iso` | Graph-similarity weighted isotropy |
| `L_rel` | Token-pair relation preservation (hinge) |
| `L_single` | Standard InfoNCE (prevents forgetting) |

---

## Experimental Results

### Embedding Geometry (Table 1)

| Model | IPS ↓ | Random-pair sim. ↓ |
|-------|:---:|:---:|
| CLIP ViT-B/32 | 0.7822 | 0.6844 |
| CLIP ViT-B/32 (LAION-2B) | 0.6406 | 0.4887 |
| NegCLIP | 0.7829 | 0.6988 |
| BLIP v1 | 0.5961 | 0.4571 |
| **GS-CLIP (Stage I)** | **0.379** | **0.2647** |

### ARO Benchmark (Table 4)

| Method | VG-Rel | Δ vs. CLIP |
|--------|:---:|:---:|
| CLIP ViT-B/32 | 59.9 | — |
| NegCLIP | 73.6 | +13.7 |
| TripletCLIP | 74.1 | +14.2 |
| CLIC-COCO | 74.3 | +14.4 |
| CLIC-RedCaps | 76.2 | +16.3 |
| **GS-CLIP** | **88.95** | **+29.05** |

### Per-Predicate IPS Breakdown

| Predicate | CLIP | GS-CLIP | Δ |
|-----------|:---:|:---:|:---:|
| on | 0.81 | 0.25 | +0.56 |
| in | 0.80 | 0.20 | +0.60 |
| behind | 0.78 | 0.25 | +0.53 |
| next to | 0.77 | 0.24 | +0.53 |
| holding | 0.82 | 0.48 | +0.34 |
| riding | 0.87 | 0.45 | +0.42 |

---

## Limitations

- Relies on dependency parsing; unusual caption structures may produce incomplete graphs.
- Visual Genome annotations contain noise that may bias Stage I.
- Cross-attention and GNN add inference overhead proportional to graph size.
- Vision encoder is kept frozen; joint adaptation could further improve grounded reasoning.

---

## Citation

```bibtex
@inproceedings{gsclip2026,
  title     = {Graph-Supervised Adaptation of {CLIP} Improves Relational Compositionality
               by Reshaping Language Embedding Geometry},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  note      = {Paper \#13326},
}
```

---

## References

Key papers this work builds on:
- CLIP: Radford et al., ICML 2021
- LoRA: Hu et al., ICLR 2022
- ARO / NegCLIP: Yuksekgonul et al., ICLR 2023
- SugarCrepe: Hsieh et al., NeurIPS 2023
- CLIC: Peleg et al., arXiv 2023
- Visual Genome: Krishna et al., IJCV 2017
