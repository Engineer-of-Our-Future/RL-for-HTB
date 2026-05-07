"""Evaluation CLI - load a checkpoint, run the suite, print metrics + dump JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from htbrl.env.stub_env import StubPentestEnv
from htbrl.eval.harness import run_suite
from htbrl.model.policy import ActorCriticPolicy, PolicyConfig
from htbrl.tokenizer.bpe import ByteLevelBPE
from htbrl.tools.loader import load_registry


def _stub_env_factory(vocab, seed: int):
    def _make():
        return StubPentestEnv(vocab, max_steps=20, seed=seed)
    return _make


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, default=None,
                   help="ByteLevelBPE saved JSON; if omitted, a fresh untrained byte-level BPE is used")
    p.add_argument("--n-envs", type=int, default=5, help="number of stub-env factories to use")
    p.add_argument("--n-episodes-per-env", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax sampling instead of multinomial")
    p.add_argument("--output-json", type=Path, default=None,
                   help="if set, dump SuiteResult.summary_dict() here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    vocab = load_registry()
    tokenizer = (
        ByteLevelBPE.load(args.tokenizer_path) if args.tokenizer_path else ByteLevelBPE.initialize()
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = PolicyConfig(**cfg_dict) if cfg_dict else PolicyConfig(
        vocab_size=tokenizer.vocab_size, n_tools=vocab.n_tools, max_seq_len=args.max_seq_len
    )
    policy = ActorCriticPolicy(cfg)
    policy.load_state_dict(ckpt["model"])
    policy.to(args.device)

    factories = [_stub_env_factory(vocab, seed=i) for i in range(args.n_envs)]
    suite = run_suite(
        env_factories=factories,
        policy=policy,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        n_episodes_per_env=args.n_episodes_per_env,
        max_steps_per_episode=args.max_steps,
        deterministic=args.deterministic,
        device=args.device,
    )

    summary = suite.summary_dict()
    print(json.dumps(summary, indent=2, default=str))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
