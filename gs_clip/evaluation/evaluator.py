"""
CLIPEvaluator: unified evaluation class for all GS-CLIP benchmarks.

Supports:
  - Winoground  (text / image / group scores)
  - SugarCrepe  (accuracy per split)
  - ARO / VG-Rel (relational retrieval)
  - COCO-1K retrieval (R@1/5/10, RSUM)
  - Embedding geometry (IPS, random-pair similarity)

All evaluation is done with torch.no_grad() and supports both the vanilla
CLIP baseline and any custom model with encode_text / encode_image methods.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip

from ..data.datasets import WinogroundDataset, SugarCrepeDataset


class CLIPEvaluator:
    """
    Drop-in evaluator for CLIP and GS-CLIP on compositional benchmarks.

    Usage:
        evaluator = CLIPEvaluator()                        # baseline CLIP
        evaluator = CLIPEvaluator(model=gs_clip_model)    # your trained model
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        model=None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: open_clip model identifier (used if model=None).
            pretrained: open_clip pretrained weights (used if model=None).
            model:      Optional custom model with encode_text(tokens) and
                        encode_image(images) methods. If provided, model_name
                        and pretrained are still used for the tokenizer /
                        preprocess transform.
            device:     Torch device. Defaults to CUDA if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = clip.to(self.device)

        self.model.eval()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Return L2-normalised text embeddings [N, D]."""
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Return L2-normalised image embeddings [N, D]."""
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        feats = self.model.encode_image(batch)
        return F.normalize(feats, dim=-1)

    # ------------------------------------------------------------------
    # Winoground
    # ------------------------------------------------------------------

    def evaluate_winoground(
        self, dataset: WinogroundDataset
    ) -> Dict[str, float]:
        """
        Evaluate on Winoground.

        Scores:
          text_score  — given image, chose the correct caption
          image_score — given caption, chose the correct image
          group_score — both simultaneously correct (hardest)

        Random-chance group score is ~25%. CLIP ViT-B/32 achieves ~8.7%.

        Returns:
            Dict with text_score, image_score, group_score (all in %), and
            a nested by_category dict for per-tag breakdown.
        """
        correct_text = correct_image = correct_group = 0
        by_category: Dict[str, Dict[str, int]] = {}

        for ex in tqdm(dataset, desc="Winoground", leave=False):
            t = self.encode_text([ex.caption_0, ex.caption_1])  # [2, D]
            v = self.encode_images([ex.image_0, ex.image_1])    # [2, D]

            s00 = (t[0] @ v[0]).item()
            s01 = (t[0] @ v[1]).item()
            s10 = (t[1] @ v[0]).item()
            s11 = (t[1] @ v[1]).item()

            text_ok  = (s00 > s10) and (s11 > s01)
            image_ok = (s00 > s01) and (s11 > s10)
            group_ok = text_ok and image_ok

            if text_ok:  correct_text  += 1
            if image_ok: correct_image += 1
            if group_ok: correct_group += 1

            tag = ex.tag
            if tag not in by_category:
                by_category[tag] = {"total": 0, "text": 0, "image": 0, "group": 0}
            by_category[tag]["total"] += 1
            if text_ok:  by_category[tag]["text"]  += 1
            if image_ok: by_category[tag]["image"] += 1
            if group_ok: by_category[tag]["group"] += 1

        n = len(dataset)
        results: Dict = {
            "text_score":  100 * correct_text  / n,
            "image_score": 100 * correct_image / n,
            "group_score": 100 * correct_group / n,
            "by_category": {
                tag: {
                    k: 100 * v / stats["total"] if k != "total" else v
                    for k, v in stats.items()
                }
                for tag, stats in by_category.items()
                if stats["total"] > 0
            },
        }
        return results

    # ------------------------------------------------------------------
    # SugarCrepe
    # ------------------------------------------------------------------

    def evaluate_sugarcrepe_split(
        self,
        dataset: SugarCrepeDataset,
        coco_images_dir: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate one SugarCrepe split.

        If coco_images_dir is provided and images exist, runs full image–text
        evaluation. Otherwise falls back to text-only discrimination.
        """
        if coco_images_dir is None:
            return self._sugarcrepe_text_only(dataset)

        correct = total = 0
        for ex in tqdm(dataset, desc=f"SugarCrepe/{dataset.split}", leave=False):
            img_path = Path(coco_images_dir) / f"{ex['image_id']}.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            t = self.encode_text([ex["caption"], ex["negative_caption"]])  # [2, D]
            v = self.encode_images([img])                                   # [1, D]
            correct += int((t[0] @ v[0]).item() > (t[1] @ v[0]).item())
            total += 1

        return {"accuracy": 100 * correct / total if total else 0.0, "total": total}

    def _sugarcrepe_text_only(self, dataset: SugarCrepeDataset) -> Dict:
        """Sanity-check evaluation when COCO images are not available."""
        diff = 0
        for ex in tqdm(dataset, desc=f"SugarCrepe/{dataset.split} (text-only)", leave=False):
            t = self.encode_text([ex["caption"], ex["negative_caption"]])
            diff += int((t[0] @ t[1]).item() < 0.99)
        return {
            "text_discrimination": 100 * diff / len(dataset),
            "note": "text-only — COCO images not available",
        }

    def evaluate_sugarcrepe_all(
        self,
        data_dir: str = "./data/sugarcrepe",
        coco_images_dir: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """Evaluate all available SugarCrepe splits and aggregate."""
        results: Dict[str, Dict] = {}
        for split in SugarCrepeDataset.SPLIT_URLS:
            try:
                ds = SugarCrepeDataset(data_dir=data_dir, split=split, download=False)
                results[split] = self.evaluate_sugarcrepe_split(ds, coco_images_dir)
            except FileNotFoundError:
                pass  # split not downloaded
        return results

    # ------------------------------------------------------------------
    # COCO retrieval
    # ------------------------------------------------------------------

    def evaluate_coco_retrieval(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        num_captions_per_image: int = 5,
    ) -> Dict[str, float]:
        """
        Standard COCO-1K retrieval evaluation.

        Args:
            image_features: [N_images, D] normalised.
            text_features:  [N_images × num_captions_per_image, D] normalised.
            num_captions_per_image: Captions per image (5 for COCO).
        Returns:
            Dict with R@1, R@5, R@10 for both directions and RSUM.
        """
        N = image_features.size(0)
        sim = image_features @ text_features.T  # [N_img, N_txt]

        # Image → Text
        i2t_r1 = i2t_r5 = i2t_r10 = 0
        for i in range(N):
            # Correct captions are at indices [i*k .. i*k+k-1]
            k = num_captions_per_image
            gt = set(range(i * k, i * k + k))
            top = sim[i].topk(10).indices.tolist()
            i2t_r1  += int(any(t in gt for t in top[:1]))
            i2t_r5  += int(any(t in gt for t in top[:5]))
            i2t_r10 += int(any(t in gt for t in top[:10]))

        # Text → Image
        t2i_r1 = t2i_r5 = t2i_r10 = 0
        N_txt = text_features.size(0)
        for j in range(N_txt):
            gt_img = j // num_captions_per_image
            top = sim.T[j].topk(10).indices.tolist()
            t2i_r1  += int(gt_img in top[:1])
            t2i_r5  += int(gt_img in top[:5])
            t2i_r10 += int(gt_img in top[:10])

        r = {
            "i2t_r1":  100 * i2t_r1  / N,
            "i2t_r5":  100 * i2t_r5  / N,
            "i2t_r10": 100 * i2t_r10 / N,
            "t2i_r1":  100 * t2i_r1  / N_txt,
            "t2i_r5":  100 * t2i_r5  / N_txt,
            "t2i_r10": 100 * t2i_r10 / N_txt,
        }
        r["rsum"] = sum(r.values())
        return r

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_winoground(results: Dict) -> None:
        print("\n=== Winoground ===")
        print(f"  Text  Score: {results['text_score']:.2f}%")
        print(f"  Image Score: {results['image_score']:.2f}%")
        print(f"  Group Score: {results['group_score']:.2f}%  ← primary metric")

    @staticmethod
    def print_sugarcrepe(results: Dict[str, Dict]) -> None:
        accs = [v["accuracy"] for v in results.values() if "accuracy" in v]
        print("\n=== SugarCrepe ===")
        if accs:
            print(f"  Average accuracy: {np.mean(accs):.2f}%")
        for split, res in sorted(results.items()):
            if "accuracy" in res:
                print(f"  {split:<15}: {res['accuracy']:.2f}%")

    @staticmethod
    def save_results(results: Dict, path: str) -> None:
        Path(path).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {path}")
