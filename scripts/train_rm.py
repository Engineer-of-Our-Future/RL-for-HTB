"""Reward-model training (Phase 8).

Reads pairwise preferences from the sqlite store, encodes the snippets via the
project's BPE tokenizer, and trains the RewardModel via Bradley-Terry loss.

    python scripts/train_rm.py \\
        --db data/preferences.db \\
        --tokenizer-path tokenizer/v1.json \\
        --epochs 8 --batch-size 16 \\
        --output checkpoints/rm-v1.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from htbrl.data.preference_dataset import PreferenceStore
from htbrl.rm.model import (
    RewardModel,
    RewardModelConfig,
    bradley_terry_loss,
    pairwise_accuracy,
)
from htbrl.tokenizer.bpe import ByteLevelBPE


class PreferenceTrainDataset(Dataset):
    """Encodes (preferred, other) snippet pairs into tokenized tensor pairs."""

    def __init__(
        self,
        store: PreferenceStore,
        tokenizer: ByteLevelBPE,
        max_seq_len: int = 1024,
        exclude_ties: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.special_id("<pad>")
        self._pairs: list[tuple[list[int], list[int]]] = []
        for preferred, other, label in store.iter_training_pairs(
            exclude_ties=exclude_ties
        ):
            p_ids = self._encode(preferred.content_text)
            o_ids = self._encode(other.content_text)
            self._pairs.append((p_ids, o_ids))

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_bos_eos=True)
        if len(ids) > self.max_seq_len:
            # Keep the tail (most recent context).
            ids = ids[-self.max_seq_len:]
        # Pad to max_seq_len for fixed-shape batching.
        return ids + [self.pad_id] * (self.max_seq_len - len(ids))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict:
        p_ids, o_ids = self._pairs[idx]
        p = torch.tensor(p_ids, dtype=torch.int64)
        o = torch.tensor(o_ids, dtype=torch.int64)
        p_mask = (p != self.pad_id)
        o_mask = (o != self.pad_id)
        return {"p": p, "p_mask": p_mask, "o": o, "o_mask": o_mask}


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/preferences.db")
    p.add_argument("--tokenizer-path", required=True, type=Path)

    # Model
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5.0e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)

    print(f"=== RM training ===")
    print(f"  db          : {args.db}")
    print(f"  tokenizer   : {args.tokenizer_path}")
    print(f"  device      : {args.device}")

    tokenizer = ByteLevelBPE.load(args.tokenizer_path)
    store = PreferenceStore(args.db)
    full_ds = PreferenceTrainDataset(store, tokenizer, max_seq_len=args.max_seq_len)
    n_total = len(full_ds)
    if n_total < 2:
        print(f"ERROR: need >= 2 preference pairs to train; have {n_total}", file=sys.stderr)
        return 2

    n_val = max(1, int(round(n_total * args.val_fraction)))
    val_ds, train_ds = torch.utils.data.random_split(
        full_ds, [n_val, n_total - n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  pairs       : {n_total} (train={len(train_ds)}, val={len(val_ds)})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate, drop_last=False,
    )

    cfg = RewardModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        d_ff=args.d_ff, max_seq_len=args.max_seq_len, dropout=args.dropout,
    )
    rm = RewardModel(cfg).to(args.device)
    optim = torch.optim.AdamW(rm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"  rm params   : {rm.n_parameters():,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    for epoch in range(args.epochs):
        # Train
        rm.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            p = batch["p"].to(args.device)
            o = batch["o"].to(args.device)
            pm = batch["p_mask"].to(args.device)
            om = batch["o_mask"].to(args.device)
            score_p = rm(p, pm)
            score_o = rm(o, om)
            loss = bradley_terry_loss(score_p, score_o)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), args.gradient_clip)
            optim.step()
            total_loss += float(loss)
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Val
        rm.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            v_n = 0
            for batch in val_loader:
                p = batch["p"].to(args.device)
                o = batch["o"].to(args.device)
                pm = batch["p_mask"].to(args.device)
                om = batch["o_mask"].to(args.device)
                score_p = rm(p, pm)
                score_o = rm(o, om)
                loss = bradley_terry_loss(score_p, score_o)
                val_loss += float(loss)
                val_acc += pairwise_accuracy(score_p, score_o)
                v_n += 1
            val_loss /= max(v_n, 1)
            val_acc /= max(v_n, 1)

        elapsed = time.time() - t0
        h = {
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc,
            "elapsed_s": round(elapsed, 1),
        }
        history.append(h)
        print(
            f"  epoch {epoch+1:>2}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  ({elapsed:.1f}s)"
        )

    torch.save({
        "model": rm.state_dict(),
        "config": asdict(cfg),
        "history": history,
        "args": vars(args),
    }, args.output)
    print(f"  saved -> {args.output}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
