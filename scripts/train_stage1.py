"""
Stage I Training: Visual Genome scene-graph pre-training.

Trains the GNN, cross-attention fusion module, and LoRA adapters on
Visual Genome relationship triples using the standard CLIP contrastive loss.
The CLIP vision encoder and the base text-encoder weights remain frozen.

Usage:
    python scripts/train_stage1.py
    python scripts/train_stage1.py --config configs/default.yaml --max_steps 10000
    python scripts/train_stage1.py --vg_root /path/to/visual_genome --batch_size 32
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gs_clip.model import GSCLIPModel
from gs_clip.data.datasets import VGSceneGraphDataset
from gs_clip.training.losses import contrastive_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_collate_fn(tokenizer, device):
    """Return a collate function that tokenises captions in the dataloader."""
    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        captions = [b["caption"] for b in batch]
        tokens = tokenizer(captions)
        return {
            "images":   images.to(device),
            "tokens":   tokens.to(device),
            "captions": captions,
        }
    return collate


def parse_args():
    p = argparse.ArgumentParser(description="GS-CLIP Stage I: Visual Genome training")
    p.add_argument("--vg_root",      default="data/visual_genome")
    p.add_argument("--max_samples",  type=int, default=50_000)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_steps",    type=int, default=5_000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--lora_rank",    type=int, default=16)
    p.add_argument("--lora_alpha",   type=float, default=32.0)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint_dir", default="checkpoints/stage1")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--log_every",    type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"VG root: {args.vg_root}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GSCLIPModel(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
    ).to(args.device)

    counts = model.count_trainable()
    total_trainable = sum(counts.values())
    print(f"\nTrainable parameters:")
    for k, v in counts.items():
        print(f"  {k:<15}: {v:>10,}")
    print(f"  {'TOTAL':<15}: {total_trainable:>10,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = VGSceneGraphDataset(
        vg_root=args.vg_root,
        preprocess=model.preprocess,
        tokenizer=model.tokenizer,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=build_collate_fn(model.tokenizer, args.device),
        drop_last=True,
        pin_memory=(args.device == "cuda"),
    )
    print(f"\nDataset: {len(dataset):,} samples | {len(loader)} batches/epoch")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step = 0
    best_loss = float("inf")
    history = []

    print(f"\n{'Step':>6}  {'Loss':>9}  {'LR':>10}")
    print("-" * 32)

    for epoch in range(9999):
        if step >= args.max_steps:
            break

        for batch in loader:
            if step >= args.max_steps:
                break

            with torch.cuda.amp.autocast(enabled=(args.device == "cuda")):
                img_feats, txt_feats = model(
                    batch["images"], batch["tokens"], batch["captions"]
                )
                loss = contrastive_loss(img_feats, txt_feats)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            loss_val = loss.item()
            history.append({"step": step, "loss": loss_val})

            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(
                    {"step": step, "model_state": model.state_dict(), "loss": loss_val},
                    ckpt_dir / "best.pt",
                )

            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"{step:>6}  {loss_val:>9.4f}  {lr_now:>10.2e}")

    # Final checkpoint
    torch.save(
        {"step": step, "model_state": model.state_dict(), "loss": best_loss},
        ckpt_dir / "final.pt",
    )
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nStage I done — {step} steps | best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to {ckpt_dir}/final.pt")
    print("\nRun Stage II next:")
    print(f"  python scripts/train_stage2.py --stage1_ckpt {ckpt_dir}/final.pt")


if __name__ == "__main__":
    main()
