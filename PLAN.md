# Master Plan — Pure-PyTorch RL Pentesting Agent for HackTheBox

## Context

You want an RL agent that learns to operate inside a HackTheBox-style attacker environment, builds skill by *doing* (interacting with target VMs over the network), and is shaped by *user feedback* (RLHF-style) on top of automated environment rewards. Constraints you set:

- **No pretrained models, no LLMs, no third-party AI weights.** Every parameter is initialized fresh and trained from your data.
- **Pure PyTorch.** PyTorch + NumPy + standard infra libs only. Transformer, tokenizer, PPO, GAE, reward model, KL controller — all from scratch.
- **Has its own critic.** Actor-critic, explicit value head, GAE returns.
- **Hardware:** RTX 3060 12 GB · 16 GB system RAM · Ryzen 7 5700G (8C/16T).

Architectural decisions you confirmed up front:
1. **Structured tool vocabulary** (~150–300 typed tool templates), not free-form token-by-token shell generation. Keeps the action space tractable on a 3060.
2. **BC warm-start *plus* cold-start RL spirit.** Pretrain on demonstrations, then PPO with curiosity (RND) + high entropy bonus so the agent keeps exploring beyond the demo distribution.
3. **Agent + PyTorch on Windows host (uses 3060 natively); Kali attacker in WSL2 / VM over SSH; HTB targets reached via OpenVPN from inside Kali.** Cleanest GPU path, lowest virtualization overhead.

The plan is a **16-week, 12-phase build**, ordered so each phase produces something runnable end-to-end. Earlier phases use stub components for everything they don't own yet, so you can integration-test continuously instead of bigbang at week 16.

---

## Hardware budget (upfront sanity check)

This drives every model-size and batch-size decision below.

| Resource              | Budget                                                  | Spent on                                                                                                      |
| --------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **VRAM 12 GB**        | ~10.5 GB usable (rest = CUDA context + driver)          | Policy (≈1.0 GB fp32) + Critic (shared trunk) + Reference policy (≈1.0 GB, frozen) + Reward model (≈0.4 GB) + AdamW state (≈2.0 GB, 8-bit) + activations + rollout cache (≈4 GB) |
| **System RAM 16 GB**  | ~12 GB usable                                           | Replay/rollout buffers (3–4 GB), tokenizer + vocab tables, demo trajectories, env subprocesses                 |
| **CPU 16 logical**    | 12 workers for env rollout (4 reserved for OS + driver) | Parallel SSH-attached envs sample trajectories while GPU does forward passes / training                       |

**Target policy size: ≈50–80 M params.**
Concretely: transformer encoder, `d_model=384`, `n_layers=8`, `n_heads=8`, `d_ff=1536`, `max_seq=1024`, vocab ≈ 32 k. Two heads (tool-id categorical, parameter-emission) + critic head share the trunk. This fits comfortably under 12 GB even with reference + reward model resident, using bf16 forward + fp32 optimizer master weights + 8-bit optimizer states.

If a phase blows the budget, fix is always one of: shrink `d_model`, gradient checkpointing on transformer blocks, drop `max_seq`, or reduce concurrent rollout workers — in that order.

---

## High-level architecture

```
┌─────────────────────────────── Windows host (3060 + 16 GB RAM) ───────────────────────────────┐
│                                                                                                │
│  ┌────────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌─────────────────┐   │
│  │ Tokenizer (BPE)│ →  │  State encoder   │ →  │  Policy + Critic  │ →  │  Action decoder │   │
│  │  trained from  │    │  (Transformer    │    │  (actor-critic    │    │  (tool-id +     │   │
│  │   shell corpus │    │   encoder, ours) │    │   heads, ours)    │    │   typed params) │   │
│  └────────────────┘    └──────────────────┘    └───────────────────┘    └────────┬────────┘   │
│         ▲                                              ▲                          │            │
│         │                                              │                          ▼            │
│  ┌────────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌─────────────────┐   │
│  │  Trajectory    │    │  Reward model    │    │  PPO trainer +    │    │  Env interface  │   │
│  │  + replay buf  │    │  (Bradley-Terry, │    │  GAE + KL ctrl +  │    │  (Gymnasium-    │   │
│  │  (RAM, mmap)   │    │   ours)          │    │  RND curiosity)   │    │   compatible)   │   │
│  └────────────────┘    └──────────────────┘    └───────────────────┘    └────────┬────────┘   │
│         ▲                       ▲                                                 │            │
│         │                       │                                                 │ SSH        │
│  ┌──────┴──────┐         ┌──────┴───────┐                                         ▼            │
│  │  Feedback   │         │  Preference  │                            ┌─────────────────────┐   │
│  │  web UI     │ ←labels │  dataset     │                            │ Kali attacker (WSL2)│   │
│  │ (FastAPI)   │         │  (sqlite)    │                            │  ┌────────────────┐ │   │
│  └─────────────┘         └──────────────┘                            │  │ tool registry  │ │   │
│         ▲                                                             │  │ (nmap, gobust, │ │   │
│         │                                                             │  │  hydra, smb…)  │ │   │
└─────────┼─────────────────────────────────────────────────────────────┼──┴────────┬───────┴─┼───┘
          │                                                             │ OpenVPN   │         │
       (you)                                                            │           ▼         │
                                                                        │   ┌──────────────┐ │
                                                                        │   │ HTB target   │ │
                                                                        │   │ (or local    │ │
                                                                        │   │  vulnhub)    │ │
                                                                        │   └──────────────┘ │
                                                                        └────────────────────┘
```

---

## Phase 0 — Project bootstrap (Week 1)

**Goal:** Empty repo → working dev loop with logging, config, lint, and a one-liner that says "GPU detected, env reachable, hello world."

### Deliverables
- `RL-for-HTB/` repo layout (see below).
- `pyproject.toml` with pinned deps. **Allowed:** `torch`, `numpy`, `gymnasium`, `pexpect`, `paramiko`, `pyyaml`, `hydra-core`, `tensorboard`, `pytest`, `fastapi`, `bitsandbytes` (for 8-bit Adam — kernels only, no model weights), `tqdm`. **Forbidden:** `transformers`, `tokenizers` (HuggingFace), `trl`, `peft`, anything with pretrained weights.
- WSL2 + Kali attacker provisioned. OpenVPN config tested against HTB starting point or a local Vulnhub VM.
- `scripts/smoke_test.py`: imports torch, prints `torch.cuda.get_device_name`, opens SSH to Kali, runs `whoami`, returns 0.

### Repo layout
```
RL-for-HTB/
├── pyproject.toml
├── README.md
├── configs/                  # Hydra configs (model/, train/, env/, rm/)
├── src/htbrl/
│   ├── tokenizer/            # BPE from scratch
│   ├── model/                # transformer, heads, init
│   ├── env/                  # gymnasium wrapper, SSH interface, parsers
│   ├── tools/                # tool registry, schema, validation
│   ├── algo/                 # PPO, GAE, RND, KL controller
│   ├── data/                 # rollout buffer, demo dataset, preference dataset
│   ├── rm/                   # reward model
│   ├── feedback/             # FastAPI UI for human labeling
│   ├── eval/                 # eval harness
│   └── utils/                # logging, checkpointing, profiling
├── scripts/
│   ├── train_bc.py
│   ├── train_ppo.py
│   ├── train_rm.py
│   ├── collect_demos.py
│   ├── serve_feedback.py
│   └── eval.py
├── tests/                    # unit + integration
└── docs/
```

### Verification
`python scripts/smoke_test.py` → exits 0, logs GPU name, logs successful SSH `whoami`.

---

## Phase 1 — Action vocabulary & tool schema (Week 2)

**Goal:** Define what the agent *can do*. This is the most important design artifact in the project — it bounds reward density, exploration, and model size.

### Approach
- Author a YAML schema per tool in `src/htbrl/tools/registry/`. ~150 tools to start, expandable.
- Each entry has: tool name, category (recon/web/exploit/post-exploit/lateral/cleanup), command template with `{slot}` placeholders, slot types (enum / int range / ip / port / wordlist-id / free-string / file-path), default values, expected runtime cap, output parser ID.
- Slot types let the policy emit *typed* parameters: most slots are categorical (small vocab), only a few are free-text (filenames, custom commands).
- Validation layer: reject out-of-range params before sending to the env. This is critical — it gives the policy bounded "syntactic" guarantees and prevents wasting rollout time on broken commands.

### Critical files
- `src/htbrl/tools/schema.py` — pydantic models for tool definitions and parameter types.
- `src/htbrl/tools/registry/*.yaml` — one file per tool family.
- `src/htbrl/tools/loader.py` — loads & validates the registry at startup, builds the action embedding index.

### Action embedding scheme
- Tool ID → learned embedding (≈300 vectors of `d_model=384` ≈ 460 K params).
- Each slot value → its own learned embedding table (sized to the slot's vocab); free-text slots use the BPE tokenizer from Phase 2.
- Policy output: a tool-ID logit vector + per-tool-conditioned slot heads. Action sampling is autoregressive across slots within a tool (small loop, ≤8 slots typical).

### Verification
- `pytest tests/tools/test_registry.py`: every YAML loads, every template renders correctly with sample params, every parser parses its example output without crashing.

---

## Phase 2 — BPE tokenizer & state representation (Week 3)

**Goal:** Build the tokenizer from scratch on a shell-relevant corpus, freeze its vocab, and define the state encoder input format.

### Tokenizer
- BPE, target vocab 32 768. Trained on: Linux man pages (apt source `manpages`), `~/.bash_history` corpus from public dumps (filtered), nmap/gobuster/metasploit output samples (collected in Phase 0–1), HTB writeups (text only, from public mirrors), and CVE descriptions.
- Implementation in pure Python (BPE is a few hundred lines). One-time training is fine on CPU.
- Special tokens: `<pad> <bos> <eos> <obs> <act> <rew> <cmd> <out> <prompt> <ip> <port> <hash>`.

### State encoder input
A turn-structured token sequence:
```
<bos> <obs> tok... <act> tool=nmap port=22 <out> tok... <rew> +0.1
<obs> tok... <act> tool=gobuster wordlist=common.txt <out> tok... <rew> +0.0
... (rolling window of last K turns, K ≈ 8)
```
Truncated/right-padded to `max_seq=1024`. Out-of-window history is summarized via a learned compression vector (LSTM over discarded turns) prepended as a single soft token — gives the policy long-horizon memory without paying full attention cost.

### Critical files
- `src/htbrl/tokenizer/bpe.py` — encoder/decoder + training script.
- `src/htbrl/tokenizer/special.py` — special-token registry (single source of truth).
- `src/htbrl/data/encode_state.py` — turn-structured serializer.

### Verification
Round-trip on 10 k random shell-output samples: `decode(encode(x)) == x` byte-perfect except for explicitly-replaced rare bytes. Tokens-per-byte stays < 0.5 on shell output (sanity for compression).

---

## Phase 3 — Neural architecture (Week 4)

**Goal:** Pure-PyTorch transformer + actor-critic heads, end-to-end forward pass works with a dummy batch.

### Model
- Transformer **encoder** only (no decoder needed — we attend over the sequence and predict from the final position).
- 8 layers, `d_model=384`, 8 heads, `d_ff=1536`, GELU, pre-LayerNorm, RoPE for positional info (more robust than learned positions for variable lengths).
- Attention: start with vanilla scaled-dot-product (`F.scaled_dot_product_attention`, which already routes to FlashAttention 2 on Ampere — counts as PyTorch primitive, not an LLM lib).
- Heads (all on top of the final-position hidden state):
  - **Tool head:** linear → softmax over tool IDs.
  - **Slot head:** conditioned on the sampled tool ID embedding; linear per slot type. Autoregressive over slots within an action.
  - **Value head:** linear → scalar V(s). Uses the same trunk.
- Initialization: GPT-2-style scaled init (we replicate the formula, not the weights).

### Memory check before going further
- Forward pass on `B=8, T=1024, d=384, L=8`: ≈ 2.5 GB activations in fp32, ≈ 1.3 GB in bf16. Fine.
- Optimizer state at 100 M params with 8-bit Adam ≈ 200 MB. Fine.

### Critical files
- `src/htbrl/model/transformer.py` — block, attention, MLP, RoPE.
- `src/htbrl/model/heads.py` — tool / slot / value heads.
- `src/htbrl/model/init.py` — weight initialization.
- `src/htbrl/model/policy.py` — composes the above; provides `forward()`, `sample_action()`, `value()`.

### Verification
- `pytest tests/model/`: deterministic forward pass shape checks; gradient-flow test (loss.backward() doesn't NaN); memory test (peak VRAM < 4 GB on dummy batch).

---

## Phase 4 — Environment wrapper (Week 5)

**Goal:** Gymnasium-compatible env that an agent can `env.step(action)` against. This is where everything becomes real.

### Components
- **SSH session manager** (`paramiko` or `pexpect` for interactive sessions). Persistent shell per env instance (so `cd`, env vars, opened SOCKS proxies survive across actions). One env = one SSH session = one Kali attacker shell.
- **Action executor:** takes structured action `{tool, slots}`, renders to a bash command via the registry template, runs with timeout, captures stdout/stderr/exit-code.
- **Output parser registry:** per-tool parsers (regex-based for nmap port lists, gobuster paths, smbclient share lists, etc.) → structured features. Both raw text and parsed features go into the next observation.
- **Reward shaping primitives** (auto rewards, *not* the learned RM yet — that's Phase 8):
  - `+0.1` first time a new open port is discovered on a target.
  - `+0.2` first time a new service version is fingerprinted.
  - `+0.5` first user shell.
  - `+1.0` user flag captured (regex `[a-f0-9]{32}` written by env on read).
  - `+1.5` root/SYSTEM shell.
  - `+2.0` root flag captured.
  - `−0.01` per command (encourages efficiency).
  - `−0.1` per timed-out command.
  - `−0.5` per detected reset condition (target unreachable, defender alert if simulated).
- **Episode lifecycle:** `reset()` reverts the target to a known snapshot (HTB doesn't allow this directly — for development we use local Vulnhub/Metasploitable in `vmrun`/`virsh`; for HTB we accept episodes are non-resettable and rotate boxes).
- **Safety guardrails:** target IP allowlist (env refuses to act on anything outside the configured CIDR — prevents accidental scans of real internet during exploration); rate limiter; per-episode wall-clock cap.

### Critical files
- `src/htbrl/env/htb_env.py` — Gymnasium env class.
- `src/htbrl/env/ssh_session.py` — persistent shell manager.
- `src/htbrl/env/parsers/` — one parser per tool (or family).
- `src/htbrl/env/rewards.py` — reward primitives + flag detection.
- `src/htbrl/env/snapshot.py` — local-VM snapshot/restore (skipped for HTB).
- `configs/env/*.yaml` — per-target configs (IP, allowlist, snapshot strategy).

### Verification
- Run a hand-scripted policy that nmaps a Metasploitable VM and parses ports. Episode terminates cleanly, total reward > 0, no SSH leaks (check `who` on Kali after).
- Property test: 100 random valid actions never crash the env, never exceed memory cap.

---

## Phase 5 — Demonstration collection & BC pretraining (Weeks 6–7)

**Goal:** Get the policy from "random tool clicks" to "looks vaguely like a pentester."

### Demo collection — "Wizard mode"
- A wrapper CLI you (or a small group) use while pwning boxes manually. It launches the same env interface, but lets you type any bash; it then asks you to pick the closest action-vocab entry (or skips logging if no match). Each captured tuple: `(state, action, reward, next_state, done)`.
- Auto-suggest: as you type, it shows the top-3 vocab matches by string similarity, you press number → logged.
- Target: **800–1500 trajectories across 25–40 boxes** (mix of HTB easy/medium starting-point boxes + Vulnhub). About 10–20 hours of focused play.

### BC training
- Cross-entropy loss on `(tool, slot1, slot2, …)` given state.
- Standard tricks: label smoothing (0.05), dropout (0.1) on transformer blocks, AdamW lr 1e-4 with cosine warm-restart, batch 32 with grad-accum 4 → effective 128.
- Train **on CPU first** with a 4-layer mini-model to validate the dataloader, then full model on GPU.
- Should converge in 1–2 days on the 3060.

### Critical files
- `scripts/collect_demos.py` — wizard-mode CLI.
- `src/htbrl/data/demo_dataset.py` — disk-mmapped trajectory format (parquet or msgpack-numpy).
- `scripts/train_bc.py` — BC training loop.
- `tests/integration/test_bc_overfit.py` — overfit a single trajectory in < 2 min as a dataloader smoke test.

### Verification
- BC policy replays a held-out demo and matches the human's tool choice ≥ 60 % of steps within first 200 epochs.
- Eval episode on a fresh easy box: agent gets at least to nmap → service enumeration without falling apart. (No expectation of full pwn yet.)

---

## Phase 6 — PPO trainer + cold-start exploration (Weeks 8–9)

**Goal:** RL fine-tuning that retains exploration ("cold-start spirit" you asked for) while not destroying the BC prior.

### PPO from scratch
- Rollout buffer in pinned RAM: `(obs, action, logprob, value, reward, done)` × T_rollout × N_envs.
- N_envs = **8 parallel** (one SSH session per env), each env runs **128 steps** between updates → 1024 transitions per update.
- GAE(λ=0.95, γ=0.995). Returns and advantages computed once per rollout, advantages normalized per-batch.
- PPO clip 0.2, value clip 0.2, value-loss coeff 0.5, **entropy coeff 0.02 (high — preserves exploration)**, 4 epochs over the rollout, minibatch 256.
- Mixed precision: bf16 forward + loss in fp32 + 8-bit AdamW optimizer state.
- Gradient checkpointing on transformer blocks (cuts activation memory ~60 % for ~25 % step-time hit). Toggle via config.

### Cold-start retention via curiosity
- **RND** (Random Network Distillation) intrinsic reward: a frozen random network outputs a target embedding; a trainable predictor learns to match. Prediction error = novelty bonus. Implemented in ~50 lines, pure PyTorch. RND target frozen at init, never updated.
- Combined reward: `r = r_env + β · r_rnd`, `β` annealed from 0.5 → 0.05 over first 5 M env steps.
- This is what makes the RL "cold-start-flavored" even though we warm-started from BC: the agent is rewarded for trying state-action combos the BC distribution didn't cover.

### KL control (vs reference policy)
- Keep a frozen copy of the BC policy as `π_ref`. Add `−c · KL(π‖π_ref)` to the reward, with `c` adapted (Schulman-style) to keep KL near 0.02. Prevents catastrophic forgetting of the BC prior.

### Critical files
- `src/htbrl/algo/ppo.py` — trainer (rollout collection ↔ update loop).
- `src/htbrl/algo/gae.py` — GAE.
- `src/htbrl/algo/rnd.py` — random network distillation.
- `src/htbrl/algo/kl_ctrl.py` — adaptive KL coefficient.
- `src/htbrl/data/rollout_buffer.py` — pinned-memory ring buffer.
- `src/htbrl/utils/multi_env.py` — N parallel envs over multiprocessing (uses 8 of your 16 logical cores).

### Verification
- Train on a single fixed easy box for 24 h. Final agent solves the box (gets user flag) > 50 % of episodes. Logs show RND bonus decreasing over time, KL staying near target, entropy not collapsing below 1.0 nats.

---

## Phase 7 — Critic refinements (Week 10)

**Goal:** Make the value function reliable enough to support both PPO and downstream RLHF.

### Improvements over Phase 6 baseline
- **Target value network** (Polyak-averaged at τ=0.005), used for bootstrap targets — reduces value-overestimation in long episodes.
- **Value clipping** (already in PPO above; confirm it's pulling its weight via ablation).
- **Reward normalization:** running-mean/std on returns, scale rewards by `1/std`. Critical for stable training when env rewards span 0.01 (small probes) to 2.0 (root flag).
- **Optional dueling decomposition:** `V(s) + A(s,a) − mean_a A(s,a)`. Try in ablation; keep only if it measurably helps.

### Critical files
- `src/htbrl/model/heads.py` — extend value head for dueling option.
- `src/htbrl/algo/ppo.py` — wire target net + reward norm + value clip.
- `src/htbrl/algo/normalizer.py` — Welford-style running stats.

### Verification
- A/B vs Phase 6: value-function explained variance > 0.6 on rollouts (was ~0.3 in vanilla PPO baseline). Episode return std halves on the same training budget.

---

## Phase 8 — Reward model from human feedback (Weeks 11–12)

**Goal:** Train a learned reward model from your pairwise preferences so the agent can learn things automated rewards miss (e.g., "this command was clever / wasted enumeration / risky / noisy").

### Feedback web UI
- FastAPI + minimal HTMX (or vanilla HTML) — no React build step needed. Served on `localhost:8765`.
- Workflow: page shows two trajectory snippets (8–16 steps each) side-by-side with rendered tool calls and outputs. You click "left better / right better / tie / discard." Stored in sqlite.
- Snippets sampled actively: prioritize pairs where the *current* RM is uncertain (max entropy in BT prediction). Also include random pairs to avoid label-distribution collapse.
- Target throughput: 50–100 comparisons per sitting, 30 min/sitting, 2 sittings/week. Need ~1500 comparisons total before RM is useful.

### Reward model
- Same transformer trunk as policy but smaller (4 layers, `d_model=256`, ≈12 M params) — inits fresh, doesn't share weights with policy.
- Output: scalar `r_θ(trajectory_snippet)`.
- Loss: Bradley-Terry pairwise — `−log σ(r(snippet_pref) − r(snippet_other))`.
- Calibration: log Spearman ρ on a held-out set of human pairs. Target ρ > 0.55 before using in PPO.

### Critical files
- `src/htbrl/feedback/server.py` — FastAPI app.
- `src/htbrl/feedback/templates/` — HTMX pages.
- `src/htbrl/feedback/active_sampler.py` — uncertainty-based pair selection.
- `src/htbrl/data/preference_dataset.py` — sqlite ↔ tensor.
- `src/htbrl/rm/model.py` — RM architecture.
- `scripts/train_rm.py` — RM training loop.

### Verification
- Held-out pairwise accuracy ≥ 65 % (random = 50 %). Spearman ρ ≥ 0.55 vs human-labeled scalar ratings on a small calibration set.

---

## Phase 9 — RLHF integration (Week 13)

**Goal:** Combine env reward, RND, KL penalty, and learned RM into the final training signal.

### Reward composition
```
r_total = r_env  +  α · r_rm  +  β · r_rnd  −  c · KL(π ‖ π_ref)
```
- α annealed 0 → 0.5 over first 1 M steps after RM is ready (avoids reward-hacking the RM before it's calibrated).
- β decayed schedule from Phase 6 continues.
- c adaptive (target KL 0.02–0.05). Tighter than vanilla RLHF because RM is small and easier to game.

### Anti-reward-hacking measures
- Periodic RM refresh: every 500 k env steps, collect 200 fresh comparisons on the *current* policy's trajectories and continue-train the RM. Catches reward drift.
- RM ensemble (optional, week 14 if time permits): train 3 RMs with different seeds; use **min** of ensemble as the reward — penalizes high-variance regions where one RM is overconfident.
- KL hard cap: if KL > 0.1 for 3 consecutive updates, halt training, alert.

### Critical files
- `src/htbrl/algo/rlhf.py` — composite reward + KL controller wiring.
- `src/htbrl/rm/refresh_loop.py` — scheduled active relabeling.

### Verification
- Compare to Phase 7 agent on 20 held-out boxes: RLHF agent achieves higher *human-evaluated* score (score 5 trajectories per agent per box, blind to which is which) while keeping foothold/root rates equal or better.

---

## Phase 10 — Hardware-targeted optimization (Week 14)

**Goal:** Push throughput on the 3060 specifically. Until this phase, lean on correctness; here, lean on speed.

### Optimizations (apply in order, profile between each)
1. **`torch.compile(mode="reduce-overhead")`** on the policy and RM forward. Free 1.3–1.8× on Ampere.
2. **bf16 forward, fp32 master weights** (already in PPO, confirm).
3. **8-bit AdamW** via `bitsandbytes` for optimizer state. Saves ≈ 1.5 GB VRAM at 100 M params.
4. **Gradient checkpointing** on transformer blocks (toggle via config — only enable when VRAM-bound).
5. **Pinned-memory pinned dataloaders** for rollout buffer → GPU transfers; CUDA streams for overlap of host→device copy with compute.
6. **CPU rollout, GPU train** split: while GPU updates on rollout *N*, the 12 CPU workers are already collecting rollout *N+1*. Implemented via `torch.multiprocessing` queues.
7. **`F.scaled_dot_product_attention`** with `enable_flash=True` (FlashAttention-2 on Ampere — PyTorch native, no third-party LLM lib).
8. **Activation offloading to CPU** for the frozen reference policy (it's only used for KL — can be fp16 + lazy-loaded).
9. **Fused RoPE + attention** kernel — *only if* profiling shows attention is the bottleneck. Custom Triton kernel, ~150 lines, optional.

### Critical files
- `src/htbrl/utils/profile.py` — wraps `torch.profiler` for one-line use.
- `src/htbrl/utils/compile_guards.py` — graceful fallback if `torch.compile` errors on a layer.
- `configs/train/optim.yaml` — toggle each optimization independently.

### Verification
- Documented before/after: target ≥ 2.5× rollout throughput and ≥ 2× update throughput vs Phase 9 baseline. Peak VRAM stays < 11 GB.

---

## Phase 11 — Evaluation harness (Week 15)

**Goal:** Rigorous, automated evaluation. No more "looks good in tensorboard" judgments.

### Eval suite
- 15 held-out boxes (10 easy, 5 medium) — *never seen during training, BC, or RM labeling*.
- Per box, 5 evaluation episodes (different starting seeds/timeouts).
- Metrics:
  - **Foothold rate:** % of episodes where user shell achieved.
  - **Root rate:** % of episodes where root/SYSTEM achieved.
  - **User-flag-time / root-flag-time:** median wall-clock to capture (episodes that fail count as cap+1).
  - **Command efficiency:** flags-per-action.
  - **KL-from-BC:** how far the policy drifted.
  - **Vocab coverage:** % of tools used at least once across all eval episodes (low coverage → policy collapse).
- Regression tests: gate any merge to `main` on no metric regressing > 5 % vs the previous tagged release.

### Critical files
- `src/htbrl/eval/harness.py` — eval runner.
- `src/htbrl/eval/metrics.py` — metric definitions.
- `scripts/eval.py` — CLI: `python scripts/eval.py --checkpoint ckpt/v3.pt --suite eval_v1.yaml`.
- `tests/regression/` — pinned eval baselines.

### Verification
Manual: run eval on Phase 9 checkpoint, confirm metrics are computed and CSV written. Then on Phase 10 checkpoint, confirm regression test gate works.

---

## Phase 12 — Polish, packaging, docs (Week 16)

**Goal:** Make the project re-runnable by future-you.

### Tasks
- Hydra-compose every config so a single command (`python scripts/train_ppo.py +experiment=v1_rlhf`) reproduces a run.
- Checkpoint registry: each saved checkpoint stores hyperparams, git SHA, eval metrics, RM version, BC version. Self-describing artifacts.
- README sections: hardware notes, install, "smoke test in 5 min", "full pipeline in 16 weeks", "common failures and fixes."
- Optional: dockerize the Kali attacker side.

### Verification
Wipe `~/.cache/htbrl` and a virtualenv, follow the README from scratch, reach a passing smoke test in < 30 minutes.

---

## Critical files reference (final)

These are the files most likely to need iteration; biased toward the algorithmic core:

- `src/htbrl/model/transformer.py`, `model/policy.py`, `model/heads.py`
- `src/htbrl/algo/ppo.py`, `algo/gae.py`, `algo/rnd.py`, `algo/kl_ctrl.py`, `algo/rlhf.py`
- `src/htbrl/env/htb_env.py`, `env/ssh_session.py`, `env/rewards.py`
- `src/htbrl/tools/registry/*.yaml`, `tools/loader.py`
- `src/htbrl/tokenizer/bpe.py`
- `src/htbrl/rm/model.py`
- `src/htbrl/feedback/server.py`
- `scripts/train_bc.py`, `train_ppo.py`, `train_rm.py`, `eval.py`

---

## Verification (end-to-end)

The plan is correct if at week 16 you can:
1. `python scripts/smoke_test.py` exits 0 (Phase 0 sanity).
2. `pytest -q` passes (per-phase unit + integration tests).
3. `python scripts/eval.py --checkpoint ckpt/final.pt --suite eval_v1.yaml` produces a CSV showing **foothold rate ≥ 60 %** and **user-flag rate ≥ 30 %** on the held-out easy boxes — without ever having pretrained on or used an external model.
4. The feedback UI at `localhost:8765` lets you label new pairs, retrain the RM in ≤ 30 min on the 3060, and the next PPO iteration uses the new RM.
5. Peak VRAM during full training stays under 11 GB; CPU saturates around 70–80 % during rollout.

---

## Risk register (so we don't fool ourselves)

- **HTB doesn't allow snapshot/reset.** Mitigation: develop on local Vulnhub/Metasploitable + custom Docker lab; HTB is for *evaluation*, not training reset loops. Phase 4 already accounts for this.
- **Demonstration data is the bottleneck.** If you can't sustain 800–1500 trajectories, BC will be weak and PPO will struggle. Mitigation: start collecting demos *during* Phase 1–2, not waiting for Phase 5.
- **Reward hacking the RM.** Mitigation: KL hard cap + scheduled RM refresh + RM-min ensemble in Phase 9.
- **3060 thermal throttling on long runs.** Mitigation: target ≤ 80 °C; cap `power_limit` via `nvidia-smi -pl` if needed; checkpoint every 100 k steps so a thermal trip costs < 30 min.
- **Action vocabulary is too narrow.** Mitigation: schema is hot-loadable; new tools can be added between training runs without retraining the whole policy (only the tool-embedding row for new tools is fresh, rest is reused — described in Phase 1's embedding scheme).

---

## Note on "save in same directory"

This plan is currently saved at `C:\Users\Hmm\.claude\plans\let-s-build-a-carefully-keen-shell.md` (required by plan mode). Once you approve the plan, I will copy it to `C:\Users\Hmm\Desktop\RL-for-HTB\PLAN.md` so it lives next to the project as you asked.
