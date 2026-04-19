"""
Evaluation script for GS-CLIP on all compositional benchmarks.

Evaluates:
  - Winoground (text / image / group score)
  - SugarCrepe (7 splits, optional COCO images)
  - Embedding geometry (IPS, random-pair similarity)

Usage:
    # Evaluate baseline CLIP
    python scripts/evaluate.py

    # Evaluate trained GS-CLIP
    python scripts/evaluate.py --checkpoint checkpoints/stage2/final.pt

    # Evaluate only Winoground
    python scripts/evaluate.py --benchmark winoground

    # Full SugarCrepe with COCO images
    python scripts/evaluate.py --coco_images /path/to/coco/val2017
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gs_clip.model import GSCLIPModel
from gs_clip.evaluation import CLIPEvaluator
from gs_clip.evaluation.metrics import intra_predicate_similarity, compute_ips_breakdown
from gs_clip.data.datasets import WinogroundDataset, SugarCrepeDataset


# Diagnostic sentence pairs for IPS measurement.
# Each tuple: (original, role-swapped) — same predicate, swapped arguments.
DIAGNOSTIC_PAIRS = {
    "on": [
        ("a cat sitting on a mat", "a mat sitting on a cat"),
        ("a book on a table", "a table on a book"),
        ("a lamp on a shelf", "a shelf on a lamp"),
    ],
    "in": [
        ("a dog in a car", "a car in a dog"),
        ("a bird in a cage", "a cage in a bird"),
        ("flowers in a vase", "a vase in flowers"),
    ],
    "wearing": [
        ("a woman wearing a hat", "a hat wearing a woman"),
        ("a man wearing glasses", "glasses wearing a man"),
    ],
    "behind": [
        ("a tree behind a house", "a house behind a tree"),
        ("a child behind a fence", "a fence behind a child"),
    ],
    "holding": [
        ("a girl holding a ball", "a ball holding a girl"),
        ("a man holding a sign", "a sign holding a man"),
    ],
    "riding": [
        ("a man riding a horse", "a horse riding a man"),
        ("a child riding a bike", "a bike riding a child"),
    ],
    "carrying": [
        ("a woman carrying a bag", "a bag carrying a woman"),
        ("a boy carrying a box", "a box carrying a boy"),
    ],
    "chasing": [
        ("a dog chasing a cat", "a cat chasing a dog"),
        ("a fox chasing a rabbit", "a rabbit chasing a fox"),
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GS-CLIP")
    p.add_argument("--checkpoint",      default=None, help="Path to model .pt file")
    p.add_argument("--benchmark",       choices=["winoground", "sugarcrepe", "geometry", "all"],
                                        default="all")
    p.add_argument("--winoground_dir",  default="data/winoground")
    p.add_argument("--sugarcrepe_dir",  default="data/sugarcrepe")
    p.add_argument("--coco_images",     default=None)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save",            default="results/json/eval_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    if args.checkpoint:
        print(f"Loading GS-CLIP from {args.checkpoint}…")
        gs_clip = GSCLIPModel(device=args.device)
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        gs_clip.load_state_dict(ckpt["model_state"])
        gs_clip.eval()
        evaluator = CLIPEvaluator(model=gs_clip, device=args.device)
        print("Custom GS-CLIP model loaded.")
    else:
        print("No checkpoint provided — evaluating baseline CLIP ViT-B/32.")
        evaluator = CLIPEvaluator(device=args.device)

    all_results = {}

    # ── Embedding geometry (IPS) ───────────────────────────────────────────────
    if args.benchmark in ("geometry", "all"):
        print("\n--- Embedding Geometry (IPS) ---")
        ips_by_pred = compute_ips_breakdown(
            model_encode_fn=evaluator.encode_text,
            pairs_by_predicate=DIAGNOSTIC_PAIRS,
            device=args.device,
        )
        print(f"  Overall IPS:  {ips_by_pred['overall']:.4f}")
        print(f"  (CLIP ViT-B/32 baseline: 0.7822 | GS-CLIP target: 0.379)")
        for pred, ips in sorted(ips_by_pred.items()):
            if pred != "overall":
                print(f"    {pred:<10}: {ips:.4f}")
        all_results["ips"] = ips_by_pred

    # ── Winoground ────────────────────────────────────────────────────────────
    if args.benchmark in ("winoground", "all"):
        print("\n--- Winoground ---")
        try:
            ds = WinogroundDataset(data_dir=args.winoground_dir, download=False)
            results = evaluator.evaluate_winoground(ds)
            evaluator.print_winoground(results)
            all_results["winoground"] = results
        except FileNotFoundError:
            print("  Winoground not found. Download with:")
            print("  python scripts/download_data.py --dataset winoground")

    # ── SugarCrepe ────────────────────────────────────────────────────────────
    if args.benchmark in ("sugarcrepe", "all"):
        print("\n--- SugarCrepe ---")
        try:
            results = evaluator.evaluate_sugarcrepe_all(
                data_dir=args.sugarcrepe_dir,
                coco_images_dir=args.coco_images,
            )
            evaluator.print_sugarcrepe(results)
            all_results["sugarcrepe"] = results
        except Exception as e:
            print(f"  SugarCrepe error: {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if all_results:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        evaluator.save_results(all_results, args.save)


if __name__ == "__main__":
    main()
