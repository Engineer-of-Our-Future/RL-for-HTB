"""Phase 5 - Behavioral cloning training script.

Trains the policy's tool head (and value head, lightly, on summed rewards as a
supervised signal) on a directory of demonstrations. The full slot-head BC is
deferred until the env layer is wired up - until then we only score the tool
choice, which is enough to validate the dataloader + training loop end-to-end.

Usage (after collecting demos under ``data/demos/`` per Phase 5):

    python scripts/train_bc.py \\
        --demo-root data/demos \\
        --tokenizer-path tokenizer/v1.json \\
        --vocab-size 32768 \\
        --epochs 5 \\
        --batch-size 32 \\
        --output checkpoints/bc-v1.pt

This script is a working skeleton: dataloader + train loop + checkpointing all
real, but it relies on a saved tokenizer (``ByteLevelBPE.save``) and a
populated ``data/demos/`` directory. Both come online after Phase 5
demonstration collection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from htbrl.algo.bc import bc_loss, tool_accuracy
from htbrl.data.demo_dataset import DemoDataset, Demonstration
from htbrl.data.encode_state import Turn, encode_episode_window
from htbrl.model.policy import ActorCriticPolicy, PolicyConfig
from htbrl.tokenizer.bpe import ByteLevelBPE
from htbrl.tools.loader import load_registry


_MATRIX_ID = {"enterprise": 0, "mobile": 1, "ics": 2}


class BCExampleDataset(Dataset):
    """Yields (token_ids, attn_mask, matrix_id, target_tool_id) tuples.

    Each demonstration is unrolled into one example per turn, with the state
    encoded as the rolling window of all turns up to (but not including) the
    one being predicted.
    """

    def __init__(
        self,
        demo_dataset: DemoDataset,
        tokenizer: ByteLevelBPE,
        max_seq_len: int,
        tool_id_by_name: dict[str, int],
        history_window: int = 8,
    ) -> None:
        self.demo_dataset = demo_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tool_id_by_name = tool_id_by_name
        self.history_window = history_window
        self._index: list[tuple[int, int]] = []  # (demo_idx, turn_idx)
        for di, demo in enumerate(demo_dataset):
            for ti in range(demo.n_turns):
                if demo.turns[ti].action_tool_name in tool_id_by_name:
                    self._index.append((di, ti))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        di, ti = self._index[idx]
        demo = self.demo_dataset[di]
        # State = turns 0..ti-1; target = turn ti's action.
        history = demo.turns[max(0, ti - self.history_window) : ti]
        history_objs = [
            Turn(
                obs_text=t.obs_text,
                action_tool_name=t.action_tool_name,
                action_render=t.action_render,
                reward=t.reward,
            )
            for t in history
        ]
        ids, mask = encode_episode_window(
            self.tokenizer, demo.matrix, history_objs, max_seq_len=self.max_seq_len
        )
        target_tool_id = self.tool_id_by_name[demo.turns[ti].action_tool_name]
        return {
            "obs_tokens": ids,
            "attn_mask": mask,
            "matrix_id": torch.tensor(_MATRIX_ID[demo.matrix], dtype=torch.int64),
            "target_tool_id": torch.tensor(target_tool_id, dtype=torch.int64),
        }


def _collate(batch):
    return {
        "obs_tokens": torch.stack([b["obs_tokens"] for b in batch]),
        "attn_mask": torch.stack([b["attn_mask"] for b in batch]),
        "matrix_id": torch.stack([b["matrix_id"] for b in batch]),
        "target_tool_id": torch.stack([b["target_tool_id"] for b in batch]),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--demo-root", type=Path, required=True, action="append",
        help=("dir containing *.msgpack(.gz) demos. Pass multiple times to "
              "merge several roots, e.g. ``--demo-root data/auto_demos "
              "--demo-root data/demos`` to mix Phase 5b academy demos with "
              "manually-collected lab walks."),
    )
    p.add_argument("--tokenizer-path", type=Path, required=True, help="ByteLevelBPE saved JSON")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1536)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, required=True, help="checkpoint output path (.pt)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--max-steps", type=int, default=None, help="cap total update steps (debugging)")
    p.add_argument("--matrix", default=None, help="restrict training to one matrix")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)

    print(f"=== BC training ===")
    for r in args.demo_root:
        print(f"  demo_root      : {r}")
    print(f"  tokenizer_path : {args.tokenizer_path}")
    print(f"  device         : {args.device}")

    # Tools + tokenizer
    vocab = load_registry()
    tool_id_by_name = {t.name: vocab.id_of(t.name) for t in vocab.tools}
    tokenizer = ByteLevelBPE.load(args.tokenizer_path)
    print(f"  tokenizer vocab_size = {tokenizer.vocab_size}")
    print(f"  registry tools       = {vocab.n_tools}")

    # Dataset (one DemoDataset per --demo-root; merge their paths so a
    # mixed BC run reads from auto_demos + demos in one pass).
    sub_datasets = [DemoDataset(r) for r in args.demo_root]
    raw_ds = sub_datasets[0]
    if len(sub_datasets) > 1:
        merged_paths = []
        for ds in sub_datasets:
            merged_paths.extend(ds.paths)
        # Use object.__new__ to bypass DemoDataset.__init__'s root-existence
        # check; we just need a paths-aware iterable.
        raw_ds = DemoDataset.__new__(DemoDataset)
        raw_ds.root = sub_datasets[0].root
        raw_ds.paths = merged_paths
    if args.matrix is not None:
        raw_ds = raw_ds.filter(matrix=args.matrix)
    print(f"  loaded {len(raw_ds)} demos across {len(sub_datasets)} root(s)")

    # Phase 5b academy demos use synthetic tool names (academy_answer,
    # academy_section_read, academy_cheat_sheet, academy_module_intro,
    # academy_sandbox_cmd) that aren't in the registry - they're synthetic
    # markers, not registry-rendered shell commands. We extend
    # ``tool_id_by_name`` with whatever synthetic tools appear in the demos
    # so BC can still learn the "given this state, predict the academy
    # action class" head. The action class is meaningful: read theory vs.
    # answer vs. run sandbox cmd is the kind of decision a curriculum-
    # following agent has to make.
    extra_synthetic = sorted({
        t.action_tool_name
        for demo in raw_ds
        for t in demo.turns
        if t.action_tool_name not in tool_id_by_name
    })
    next_id = max(tool_id_by_name.values(), default=-1) + 1
    for name in extra_synthetic:
        tool_id_by_name[name] = next_id
        next_id += 1
    if extra_synthetic:
        print(f"  + {len(extra_synthetic)} synthetic tool(s) from demos: "
              f"{', '.join(extra_synthetic)}")
    n_tools_total = len(tool_id_by_name)
    print(f"  total tool head    = {n_tools_total}")

    ds = BCExampleDataset(
        raw_ds, tokenizer, max_seq_len=args.max_seq_len, tool_id_by_name=tool_id_by_name
    )
    print(f"  unrolled to {len(ds)} BC examples")

    if len(ds) == 0:
        print("ERROR: no BC examples; have any demos been collected?", file=sys.stderr)
        return 2

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate, drop_last=True,
    )

    # Model
    cfg = PolicyConfig(
        vocab_size=tokenizer.vocab_size,
        n_tools=n_tools_total,
        n_matrices=3,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=0.1,
        slot_vocab_sizes=(),
    )
    policy = ActorCriticPolicy(cfg).to(args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"  policy params = {policy.n_parameters():,}")

    # Train loop
    step = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        t0 = time.time()
        policy.train()
        for batch in loader:
            obs = batch["obs_tokens"].to(args.device)
            mask = batch["attn_mask"].to(args.device)
            mid = batch["matrix_id"].to(args.device)
            tgt = batch["target_tool_id"].to(args.device)
            tool_logits, _, _ = policy(obs, mid, mask)
            loss, metrics = bc_loss(
                tool_logits, tgt, label_smoothing=args.label_smoothing
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_acc += tool_accuracy(tool_logits, tgt)
            epoch_loss += metrics["loss/total"]
            n_batches += 1
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        print(
            f"  epoch {epoch+1}/{args.epochs}  step {step:>6}  "
            f"loss={avg_loss:.4f}  tool_acc={avg_acc:.3f}  elapsed={elapsed:.1f}s"
        )
        if args.max_steps is not None and step >= args.max_steps:
            break

    torch.save(
        {
            "model": policy.state_dict(),
            "config": cfg.__dict__,
            "tool_id_by_name": tool_id_by_name,
            "training_args": vars(args),
        },
        args.output,
    )
    print(f"  saved checkpoint -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
