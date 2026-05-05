"""Pinned-RAM ring buffer for PPO rollouts.

Holds (obs_tokens, attn_mask, matrix_id, action_tool_id, log_prob, value,
reward, done) tuples for ``T`` steps across ``N`` parallel envs. Stored on
host (pinned for fast H2D copy), batches assembled lazily during update.

Memory budget (PLAN.md "Hardware budget"):
- T=128, N=8, max_seq=1024 -> 1024 transitions per rollout
- per transition: obs_tokens int64 (1024 * 8 = 8 KB) + masks + scalars
  ~= 10 KB worst case -> ~10 MB per rollout. Trivially fits in 16 GB.

Design notes:
- We keep tensors on CPU until update; inference forward passes copy minibatches
  to GPU one at a time. This lets a 12 GB GPU train with bigger T or N than
  it could if the buffer lived on the device.
- Pinning is enabled on Linux + Windows when available; falls back gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """One minibatch returned by ``RolloutBuffer.iter_minibatches``."""

    obs_tokens: torch.Tensor       # (B, T_seq) int64
    attn_mask: torch.Tensor        # (B, T_seq) bool
    matrix_id: torch.Tensor        # (B,) int64
    action_tool_id: torch.Tensor   # (B,) int64
    log_prob_old: torch.Tensor     # (B,) fp32
    value_old: torch.Tensor        # (B,) fp32
    advantage: torch.Tensor        # (B,) fp32  (filled by GAE)
    return_: torch.Tensor          # (B,) fp32  (filled by GAE)


class RolloutBuffer:
    """Fixed-size storage for ``T * N`` transitions. Single-write per step."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        max_seq_len: int,
        device: torch.device | str = "cpu",
        pin_memory: bool = True,
    ) -> None:
        if n_steps <= 0 or n_envs <= 0 or max_seq_len <= 0:
            raise ValueError("n_steps, n_envs, max_seq_len must all be positive")
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.ptr = 0  # next write index along the time axis

        pin = pin_memory and self.device.type == "cpu" and torch.cuda.is_available()

        def _alloc(*shape, dtype):
            return torch.zeros(*shape, dtype=dtype, device=self.device, pin_memory=pin)

        # (T, N, T_seq) and (T, N) shapes
        self.obs_tokens = _alloc(n_steps, n_envs, max_seq_len, dtype=torch.int64)
        self.attn_mask = _alloc(n_steps, n_envs, max_seq_len, dtype=torch.bool)
        self.matrix_id = _alloc(n_steps, n_envs, dtype=torch.int64)
        self.action_tool_id = _alloc(n_steps, n_envs, dtype=torch.int64)
        self.log_prob = _alloc(n_steps, n_envs, dtype=torch.float32)
        self.value = _alloc(n_steps, n_envs, dtype=torch.float32)
        self.reward = _alloc(n_steps, n_envs, dtype=torch.float32)
        self.done = _alloc(n_steps, n_envs, dtype=torch.bool)

        # Filled by ``set_advantages_and_returns`` once the rollout completes.
        self.advantage = _alloc(n_steps, n_envs, dtype=torch.float32)
        self.return_ = _alloc(n_steps, n_envs, dtype=torch.float32)

    @property
    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    @property
    def total_transitions(self) -> int:
        return self.ptr * self.n_envs

    # ----- writes -------------------------------------------------------------

    def add(
        self,
        obs_tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        matrix_id: torch.Tensor,
        action_tool_id: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        if self.is_full:
            raise RuntimeError("RolloutBuffer is full; call reset() before writing more")

        i = self.ptr
        if obs_tokens.shape != (self.n_envs, self.max_seq_len):
            raise ValueError(
                f"obs_tokens shape {tuple(obs_tokens.shape)} != ({self.n_envs}, {self.max_seq_len})"
            )
        self.obs_tokens[i].copy_(obs_tokens.to(torch.int64))
        self.attn_mask[i].copy_(attn_mask.to(torch.bool))
        self.matrix_id[i].copy_(matrix_id.to(torch.int64))
        self.action_tool_id[i].copy_(action_tool_id.to(torch.int64))
        self.log_prob[i].copy_(log_prob.to(torch.float32))
        self.value[i].copy_(value.to(torch.float32))
        self.reward[i].copy_(reward.to(torch.float32))
        self.done[i].copy_(done.to(torch.bool))
        self.ptr += 1

    def set_advantages_and_returns(
        self,
        advantage: torch.Tensor,
        return_: torch.Tensor,
    ) -> None:
        if advantage.shape != (self.n_steps, self.n_envs):
            raise ValueError(
                f"advantage shape {tuple(advantage.shape)} != ({self.n_steps}, {self.n_envs})"
            )
        if return_.shape != advantage.shape:
            raise ValueError("return shape mismatch with advantage")
        self.advantage.copy_(advantage.to(torch.float32))
        self.return_.copy_(return_.to(torch.float32))

    def reset(self) -> None:
        self.ptr = 0

    # ----- reads --------------------------------------------------------------

    def iter_minibatches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
        device: torch.device | str | None = None,
    ):
        """Yield flat minibatches of ``RolloutBatch``. Lazily moves to ``device``."""
        if not self.is_full:
            raise RuntimeError(
                f"buffer not full ({self.ptr}/{self.n_steps}); call PPO update only "
                f"after collecting a full rollout"
            )
        total = self.n_steps * self.n_envs
        order = torch.randperm(total) if shuffle else torch.arange(total)
        out_device = torch.device(device) if device is not None else self.device

        # Flatten time + env axes for simple indexing.
        obs = self.obs_tokens.reshape(total, self.max_seq_len)
        mask = self.attn_mask.reshape(total, self.max_seq_len)
        mid = self.matrix_id.reshape(total)
        act = self.action_tool_id.reshape(total)
        lp = self.log_prob.reshape(total)
        val = self.value.reshape(total)
        adv = self.advantage.reshape(total)
        ret = self.return_.reshape(total)

        for start in range(0, total, minibatch_size):
            idx = order[start : start + minibatch_size]
            yield RolloutBatch(
                obs_tokens=obs[idx].to(out_device, non_blocking=True),
                attn_mask=mask[idx].to(out_device, non_blocking=True),
                matrix_id=mid[idx].to(out_device, non_blocking=True),
                action_tool_id=act[idx].to(out_device, non_blocking=True),
                log_prob_old=lp[idx].to(out_device, non_blocking=True),
                value_old=val[idx].to(out_device, non_blocking=True),
                advantage=adv[idx].to(out_device, non_blocking=True),
                return_=ret[idx].to(out_device, non_blocking=True),
            )
