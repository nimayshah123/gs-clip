"""
Compare GS-CLIP results against baseline models.

Prints a formatted table matching Table 1/4 from the paper and can
generate a LaTeX table for inclusion in the write-up.

Usage:
    python scripts/compare_baselines.py results/json/eval_results.json
    python scripts/compare_baselines.py results/json/eval_results.json --latex
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Published baseline numbers from the paper (Table 1 and Table 4)
PAPER_BASELINES = {
    "CLIP ViT-B/32":           {"ips": 0.7822, "random_sim": 0.6844, "aro_vg_rel": 59.9},
    "CLIP ViT-B/32 (LAION-2B)": {"ips": 0.6406, "random_sim": 0.4887},
    "NegCLIP":                 {"ips": 0.7829, "random_sim": 0.6988, "aro_vg_rel": 73.6},
    "TripletCLIP":             {"aro_vg_rel": 74.1},
    "CLIC-COCO":               {"aro_vg_rel": 74.3},
    "CLIC-RedCaps":            {"aro_vg_rel": 76.2},
    "GS-CLIP (ours)":          {"ips": 0.379,  "random_sim": 0.2647, "aro_vg_rel": 88.95},
}


def load(path: str) -> Dict:
    return json.loads(Path(path).read_text())


def print_geometry_table(results: Dict) -> None:
    print("\n" + "=" * 65)
    print(" Table 1 — Text Embedding Geometry")
    print("=" * 65)
    print(f"{'Model':<30} {'IPS ↓':>10} {'Random-sim ↓':>14}")
    print("-" * 55)
    for model, vals in PAPER_BASELINES.items():
        if "ips" in vals:
            print(f"{model:<30} {vals['ips']:>10.4f} {vals.get('random_sim', '—'):>14}")
    if "ips" in results:
        ips_results = results["ips"]
        your_ips = ips_results.get("overall", "?")
        print(f"\n{'Your model':<30} {your_ips:>10.4f}")
    print("=" * 65)


def print_aro_table(results: Dict) -> None:
    print("\n" + "=" * 50)
    print(" Table 4 — ARO Benchmark (VG-Relation)")
    print("=" * 50)
    print(f"{'Method':<30} {'VG-Rel':>8} {'Δ':>8}")
    print("-" * 50)
    clip_base = PAPER_BASELINES["CLIP ViT-B/32"].get("aro_vg_rel", 0)
    for model, vals in PAPER_BASELINES.items():
        if "aro_vg_rel" in vals:
            delta = vals["aro_vg_rel"] - clip_base
            sign = "+" if delta > 0 else ""
            print(f"{model:<30} {vals['aro_vg_rel']:>8.2f} {sign}{delta:>7.2f}")
    print("=" * 50)


def print_winoground_comparison(results: Dict) -> None:
    if "winoground" not in results:
        return
    wg = results["winoground"]
    print("\n" + "=" * 55)
    print(" Winoground Results")
    print("=" * 55)
    print(f"{'Metric':<20} {'CLIP (paper)':>14} {'Your model':>12}")
    print("-" * 50)
    # Published CLIP ViT-B/32 numbers from the Winoground paper
    clip_wg = {"Text Score": 31.1, "Image Score": 11.4, "Group Score": 8.7}
    mapping = {"text_score": "Text Score", "image_score": "Image Score", "group_score": "Group Score"}
    for key, label in mapping.items():
        clip_val = clip_wg[label]
        your_val = wg.get(key, "N/A")
        if isinstance(your_val, float):
            delta = your_val - clip_val
            print(f"{label:<20} {clip_val:>13.1f}% {your_val:>11.1f}%  ({delta:+.1f}%)")
        else:
            print(f"{label:<20} {clip_val:>13.1f}% {'N/A':>12}")
    print("=" * 55)


def generate_latex(results: Dict) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Embedding geometry comparison (Table 1 from paper)}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Model & IPS $\downarrow$ & Random-pair sim. $\downarrow$ \\",
        r"\hline",
    ]
    for model, vals in PAPER_BASELINES.items():
        if "ips" in vals:
            ips = f"{vals['ips']:.4f}"
            rsim = f"{vals.get('random_sim', '---')}" if isinstance(vals.get("random_sim"), float) else "---"
            lines.append(f"{model} & {ips} & {rsim} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("results_file", nargs="?", default="results/json/eval_results.json")
    p.add_argument("--latex", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    results: Dict = {}
    try:
        results = load(args.results_file)
    except FileNotFoundError:
        print(f"Results file not found: {args.results_file}")
        print("Run evaluation first: python scripts/evaluate.py")

    print_geometry_table(results)
    print_aro_table(results)
    print_winoground_comparison(results)

    if args.latex:
        print("\n" + "=" * 55)
        print(" LaTeX Table")
        print("=" * 55)
        print(generate_latex(results))


if __name__ == "__main__":
    main()
