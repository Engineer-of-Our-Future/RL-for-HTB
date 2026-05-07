# Master Plan — Pure-PyTorch RL Pentesting Agent (ATT&CK-aligned)

## Context

You want an RL agent that learns to operate inside a HackTheBox-style attacker environment, builds skill by *doing* (interacting with target VMs over the network), and is shaped by *user feedback* (RLHF-style) on top of automated environment rewards. **The agent must work across the entire MITRE ATT&CK framework — Enterprise, Mobile, and ICS matrices.** Constraints you set:

- **No pretrained models, no LLMs, no third-party AI weights.** Every parameter is initialized fresh and trained from your data.
- **Pure PyTorch.** PyTorch + NumPy + standard infra libs only. Transformer, tokenizer, PPO, GAE, reward model, KL controller — all from scratch.
- **Has its own critic.** Actor-critic, explicit value head, GAE returns.
- **MITRE ATT&CK-first.** Every tool, action, demonstration, reward signal, and eval metric is tagged with ATT&CK tactic + technique IDs. Coverage of the framework is a primary success metric, not a side effect.
- **Hardware:** RTX 3060 12 GB · 16 GB system RAM · Ryzen 7 5700G (8C/16T).

Architectural decisions you confirmed up front:
1. **Structured tool vocabulary** (~150–300 typed tool templates per matrix), not free-form token-by-token shell generation. Keeps the action space tractable on a 3060.
2. **BC warm-start *plus* cold-start RL spirit.** Pretrain on demonstrations, then PPO with curiosity (RND) + high entropy bonus so the agent keeps exploring beyond the demo distribution.
3. **Agent + PyTorch on Windows host (uses 3060 natively); attacker tooling in WSL2 / VM over SSH; targets reached via OpenVPN (HTB), Android emulator (Mobile), or local ICS lab (OpenPLC/ConPot).** Cleanest GPU path, lowest virtualization overhead.

The plan is delivered as **three release tracks** — same model, same training loop, same RLHF stack, expanded action space and env wrappers per matrix:

| Release | Weeks | Scope |
| ------- | ----- | ----- |
| **v0.1 — Enterprise** | 1–16 | The core 12-phase pipeline below. ATT&CK-Enterprise coverage, HTB + local Vulnhub. |
| **v0.2 — + Mobile**   | 17–22 | Phases 13–15. Android emulator + ADB/Frida/MobSF tool registry; Mobile-specific tactic heads. |
| **v0.3 — + ICS**      | 23–28 | Phases 16–18. OpenPLC/ConPot/GRFICSv2 lab; Modbus/S7/DNP3 tool registry; ICS tactics. |

Each track produces something runnable end-to-end at every phase. Earlier phases use stub components for everything they don't own yet, so you can integration-test continuously instead of bigbang at the end.

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

## MITRE ATT&CK alignment (cross-cutting)

The agent doesn't get a vague "be a pentester" objective — its action space, reward signal, and evaluation are all expressed in terms of MITRE ATT&CK tactics and techniques. This is what lets us measure progress objectively across very different target types (a Linux web box, an Active Directory domain, an Android device, a PLC).

### Matrices targeted

| Matrix | Track | Tactics (count) | Notes |
| ------ | ----- | --------------- | ----- |
| **Enterprise** | v0.1 (weeks 1–16) | 14 — Reconnaissance, Resource Development, Initial Access, Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, Command and Control, Exfiltration, Impact | The HTB-native matrix. Covers Windows / Linux / macOS / Cloud / SaaS / Containers / Network / IaaS sub-platforms. |
| **Mobile**     | v0.2 (weeks 17–22) | 14 — same names as Enterprise but mobile-specific techniques | Android-focused (iOS dynamic instrumentation needs jailbroken devices we won't model). Tooling: ADB, Frida, MobSF, drozer, jadx, apktool. |
| **ICS**        | v0.3 (weeks 23–28) | 12 — Initial Access, Execution, Persistence, Privilege Escalation, Evasion, Discovery, Lateral Movement, Collection, Command and Control, Inhibit Response Function, Impair Process Control, Impact | Industrial control / OT. Lab: OpenPLC, ConPot honeypot, GRFICSv2 simulator. Tools: modbus-cli, plcscan, smod, snap7-cli. |

### Cross-cutting design rules

1. **Every tool in every registry YAML carries an `attack:` block.** Required keys: `matrices` (subset of `[enterprise, mobile, ics]`), `tactics` (list of `TA0xxx` IDs), `techniques` (list of `Txxxx` or `Txxxx.xxx` sub-technique IDs). Loader rejects tools missing this block.

2. **The policy's tool head is matrix-aware.** A learned matrix-embedding (3-way) is concatenated to the trunk hidden state before the tool-logit projection. The same trunk handles all three matrices; only a small per-matrix bias adapts. This is what lets us add Mobile and ICS without retraining the trunk from scratch.

3. **Reward shaping includes a coverage bonus.** Beyond environment rewards (flag, shell, file), the reward function adds:
   - `+0.05` first time a new technique is exercised in the current episode.
   - `+0.20` first time a new tactic is *completed* (at least one technique under it succeeds).
   - These bonuses anneal over training so the agent eventually optimizes for *outcomes*, not just *techniques attempted*. Critical to prevent the agent from shallowly running every tool just for the bonus.

4. **Eval metrics are organized by matrix and tactic.** The Phase 11 eval harness reports per-matrix:
   - **Technique attempt coverage** — % of techniques tagged in the registry that the agent has at least once executed successfully across all eval episodes.
   - **Tactic completion rate** — for each tactic, % of eval episodes in which at least one technique under it succeeded.
   - **Killchain depth** — max sequential tactics chained within a single episode (e.g., Recon → Initial Access → Execution → Discovery → Privilege Escalation = depth 5).
   - **Cross-tactic transfer** — train on tactics A,B,C, hold out tactic D, measure D performance.

5. **Demonstrations are stratified by tactic.** Phase 5 demo collection requires at least 50 trajectories per tactic per matrix before BC pretraining starts. Otherwise the BC prior over-represents discovery/recon (the easy stuff) and under-represents privilege-escalation/lateral-movement.

### Out-of-scope for this plan

- Auto-generating exploits (T1588.005, T1588.006). The agent invokes existing exploits via `searchsploit` / `msfconsole`; it does not write new ones. Generating novel exploits is a separate research project.
- Adversary infrastructure (Resource Development tactic TA0042). The agent uses pre-provisioned infrastructure (Kali host); it does not register domains or compromise third-party hosts. Out of scope for ethical and legal reasons in this project.
- Real-time defender evasion against EDR. The reward function rewards *not getting blocked* but does not learn novel evasion techniques (Defense Evasion tactic TA0005 is partially covered — only via existing tools like `proxychains`, `obfuscated payloads from msfvenom`).

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

**Goal:** Define what the agent *can do* and tag it to MITRE ATT&CK. This is the most important design artifact in the project — it bounds reward density, exploration, model size, *and* coverage metrics.

### Approach
- Author YAML schemas per tool in `src/htbrl/tools/registry/`, one file per tool family. **v0.1 target: ~200 Enterprise-matrix tools** covering all 14 tactics with at least 5 distinct techniques per tactic. v0.2 adds ~80 Mobile tools, v0.3 adds ~60 ICS tools.
- Each entry has: tool name, category, command template with `{slot}` placeholders, slot types (enum / int range / ip / port / cidr / wordlist-id / free-string / file-path / hostname / hash), default values, expected runtime cap, output parser ID, `requires_root` flag, **and a required `attack:` metadata block** (matrices, tactics, techniques).
- Slot types let the policy emit *typed* parameters: most slots are categorical (small vocab), only a few are free-text (filenames, custom commands).
- Validation layer: reject out-of-range params before sending to the env. This is critical — it gives the policy bounded "syntactic" guarantees and prevents wasting rollout time on broken commands.

### ATT&CK metadata schema (required per tool)
```yaml
attack:
  matrices: [enterprise]            # subset of [enterprise, mobile, ics]
  tactics: [TA0007]                 # one or more TA0xxx IDs (Discovery here)
  techniques: [T1135, T1018]        # one or more Txxxx[.xxx] IDs
```
Loader rejects any tool missing this block. Loader exposes:
- `vocab.tools_for_technique(T1046)` — list[ToolDefinition]
- `vocab.tools_for_tactic(TA0007)` — list[ToolDefinition]
- `vocab.tools_for_matrix("enterprise")` — list[ToolDefinition]
- `vocab.coverage_summary()` — dict mapping technique ID → tool count (for "are we missing any tactics?" CI checks)

### Critical files
- `src/htbrl/tools/schema.py` — pydantic models for tool definitions, parameter types, and `AttackTags`.
- `src/htbrl/tools/registry/*.yaml` — one file per tool family. Suggested v0.1 split: `recon.yaml`, `enum.yaml`, `web.yaml`, `initial_access.yaml`, `execution.yaml`, `persistence.yaml`, `privesc.yaml`, `defense_evasion.yaml`, `credential_access.yaml`, `discovery.yaml`, `lateral_movement.yaml`, `collection.yaml`, `c2.yaml`, `exfil.yaml`, `impact.yaml`. (One file per tactic keeps reviews scoped.)
- `src/htbrl/tools/loader.py` — loads & validates the registry at startup, builds the action embedding index, exposes ATT&CK lookups.
- `src/htbrl/tools/coverage.py` — generates a coverage report (pretty-printed table + machine-readable JSON) for CI gating.

### Action embedding scheme
- **Matrix embedding:** 3 learned vectors (Enterprise, Mobile, ICS), each `d_model=384`. Selected at episode start based on the env's matrix.
- **Tool ID → learned embedding** (≈350 vectors after v0.3, `d_model=384` ≈ 540 K params).
- Each slot value → its own learned embedding table (sized to the slot's vocab); free-text slots use the BPE tokenizer from Phase 2.
- Policy output: matrix-conditioned tool-ID logit vector + per-tool-conditioned slot heads. Action sampling is autoregressive across slots within a tool (small loop, ≤8 slots typical).
- Tools tagged with multiple matrices (e.g., `nmap` works for Enterprise *and* ICS reconnaissance) appear in both tool-ID spaces but share the same embedding row — saves params and lets the trunk transfer knowledge.

### Verification
- `pytest tests/tools/test_registry.py`: every YAML loads, every template renders correctly with sample params, every tool has a valid `attack:` block, every parser parses its example output without crashing.
- `python scripts/coverage_report.py`: prints per-tactic tool count; CI fails if any Enterprise tactic has < 5 tools (configurable threshold per matrix).

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
  - **Outcome rewards** (the dominant signal):
    - `+0.1` first time a new open port is discovered on a target.
    - `+0.2` first time a new service version is fingerprinted.
    - `+0.5` first user shell.
    - `+1.0` user flag captured (regex `[a-f0-9]{32}` written by env on read).
    - `+1.5` root/SYSTEM shell.
    - `+2.0` root flag captured.
  - **ATT&CK coverage rewards** (annealed — strong early, weak late, so the agent first explores breadth then optimizes for outcomes):
    - `+0.05` first time a new technique is exercised in the current episode.
    - `+0.20` first time a new tactic is *completed* in this episode (at least one technique under it succeeds).
    - `+0.50` first time the agent chains 5+ tactics in a single episode (kill-chain depth bonus).
    - All three are scaled by `coverage_anneal(t) = max(0.1, 1.0 − t / 5_000_000)` over training steps.
  - **Penalties**:
    - `−0.01` per command (encourages efficiency).
    - `−0.1` per timed-out command.
    - `−0.5` per detected reset condition (target unreachable, defender alert if simulated).
- **ATT&CK tracking:** the env keeps a per-episode set of techniques attempted + techniques succeeded (a technique "succeeds" iff its tool returns a non-empty parsed observation, configurable per tool). Exposed in `info["attack"]` on every step for logging.
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

## Phase 5b — HTB Academy auto-learner ✅ DELIVERED (Weeks 6–8)

**Goal (delivered):** Automatically work through HTB Academy modules in tier order — read the module, harvest its cheat sheet, run the per-section sandbox, attempt the questions — and emit `Demonstration` files that BC training (Phase 5) can mix in alongside human-collected demos. **The agent does the academy's curriculum first; only after academy progression does it move to lab boxes.**

### Architecture (as built)
- **Transport:** `htbrl.academy.session.AcademySession` is the abstract transport. `MockAcademySession` ships in-tree for tests + dry-runs. The *real* transport is a Chrome DevTools Protocol attach to a user-launched Chrome (no Playwright launch markers — bypasses Cloudflare bot detection); plumbing in `htbrl.academy.cdp_walker` is the single source of truth for both single-module and multi-module walkers.
- **Answerer:** `htbrl.academy.answerer.HeuristicAnswerer` is now a *ranked-candidate* proposer:
  - `propose(question, section, *, top_n=N, module=...)` returns a list of candidates ordered by confidence, deduped by `answer_text`.
  - Per-type generators fire in parallel: acronym expansion, "how many" bullet count, port/version/path/filename extraction, inline-code-ranked (by surrounding-sentence overlap), sentence-anchor inline code, bullet-head match, quoted/backticked spans, lone-inline fallback, and **cheat-sheet-row match** (highest-precision; uses the module-level cheat sheet description→command lookup).
  - Junk filter rejects single-char / punctuation-only / English-stopword candidates so BC isn't taught to answer `'h'` / `'.'` / `'/'`.
  - Legacy `answer()` returns `propose()[0]`.
- **Cheat sheet harvest:** `cdp_walker.fetch_module_via_api(cdp, mid)` calls `GET /api/v2/modules/<id>` from inside the authenticated page; `parse_cheatsheet_markdown` turns the markdown table into structured rows (`[{"command": "ls", "description": "lists files"}, ...]`). The same fetch also yields `prelude`, `conclusion`, `takeaways`, `name`.
- **Demo turn order** (per module, BC-friendly): `academy_module_intro` (prelude+takeaways+conclusion) → `academy_cheat_sheet` (canonical command/desc table) → for each section: `academy_section_read` (theory) → `academy_answer` × N (Q&A); plus `academy_sandbox_cmd` when a flag was derived from a sandbox command. Every turn carries `techniques_attempted` / `techniques_succeeded` from `htbrl.academy.mitre_mapping`.
- **Wizard mode** (`scripts/htb_academy_wizard.py`): operator-in-the-loop. Model proposes top-N candidates per question; operator can accept top, pick alt, type custom, or skip. Auto-submit fills the input + clicks Submit via DOM; **lab-flag-shaped questions are NEVER auto-submitted** per the project rule (would risk an account ban).
- **Below-threshold path:** below `manual_review_threshold` (default 0.3) the orchestrator records a `ManualReviewItem` rather than guessing.
- Every run writes one `Demonstration` per module to `data/auto_demos/`. Synthetic tool names (`academy_module_intro`, `academy_cheat_sheet`, `academy_section_read`, `academy_answer`, `academy_sandbox_cmd`) keep these turns distinguishable from real-env demos in the BC trainer.

### Unlock gate (operator's rule)
The academy charges cubes to *open* a module, so opening a second one before the first is finished wastes the research account's budget. `curriculum.check_unlock_gate(current_module, answered_qids, cubes_before, cubes_after)` enforces *"open new module only if all questions are answered AND cube balance are updated"*. Both halves are independently toggleable. The orchestrator runs the gate before each new module open; the wizard runs it once at end-of-walk and prints a safe-to-proceed verdict.

### Modes
| Mode | Behavior | Default? |
| ---- | -------- | -------- |
| `study_only` | Read content, run sandbox, log demos. **No POSTs.** | ✅ Yes |
| `auto_submit` | Wizard fills + clicks Submit when top candidate's confidence ≥ `--auto-confidence`. Lab flags excluded. | ❌ Opt-in: `--auto-submit` |

**ToS warning.** Auto-submitting on someone's HTB Academy account to farm cubes/XP is a gray-zone use of an educational platform and may violate HTB's Terms of Service. The default is `study_only`. The submit path exists for users who explicitly accept the risk on a research account.

### Critical files (delivered)
- `src/htbrl/academy/page_models.py` — `AcademyModule` (with `cheat_sheet`, `prelude`, `conclusion`, `takeaways`, `category`, `path_ids`), `AcademySection`, `AcademyQuestion`, `AcademyAnswer`, `AcademySandbox`, `ProgressState`, `QuestionType`.
- `src/htbrl/academy/session.py` — `AcademySession` ABC + `MockAcademySession` + redacted `AcademyCredentials`.
- `src/htbrl/academy/answerer.py` — `HeuristicAnswerer` with ranked `propose()` API + cheat-sheet matcher + junk filter.
- `src/htbrl/academy/cdp_walker.py` — CDP plumbing: `open_cdp`, `fetch_module_via_api`, `parse_cheatsheet_markdown`, `enter_module`, `go_to_first_section`, `scrape_section`, `click_next_and_advance`, `submit_answer_in_dom`, `read_cube_balance`, `is_lab_flag_question`.
- `src/htbrl/academy/sandbox.py` — `SandboxRunner` (adapts `SSHSession`).
- `src/htbrl/academy/curriculum.py` — `next_module`, `check_unlock_gate`, `classify_module`, path-aware ordering.
- `src/htbrl/academy/mitre_mapping.py` — `ACADEMY_MODULE_TECHNIQUES` keyword→ATT&CK table + `techniques_for_module` / `techniques_for_section`.
- `src/htbrl/academy/orchestrator.py` — `AutoLearner` main loop with gate enforcement + per-module dedupe.
- `src/htbrl/academy/auto_demo_writer.py` — `module_intro_turn`, `cheat_sheet_turn`, `section_read_turn`, `answer_to_demo_turn`, `sandbox_cmd_to_demo_turn`, `session_to_demonstration`.
- `src/htbrl/academy/walkthrough.py` — `LabWalkthroughBuilder` (renders demos as Markdown for operator review).
- `scripts/htb_academy.py` — original CLI (mock transport).
- `scripts/htb_academy_login_check.py` — multi-mode login probe (CDP-attach, manual, no-login).
- `scripts/htb_academy_run.py` — single-module study-only walker against live academy.
- `scripts/htb_academy_wizard.py` — single-module operator-in-the-loop walker with optional auto-submit.
- `scripts/htb_academy_walk_all.py` — multi-module driver: walks every owned/in_progress module, gate-checked.
- `scripts/htb_academy_list_modules.py` — discovery CLI (table of owned/in_progress/locked + walked status).
- `scripts/academy_coverage.py` — per-demo coverage report (sections, questions, theory KB, MITRE techniques, method tags).
- `scripts/start_chrome_for_htb.ps1` — launches user's Chrome with `--remote-debugging-port=9222`.
- `configs/academy/academy_default.yaml` — Hydra config.

### Progression rule (delivered)
1. **Modules first.** The agent walks the academy curriculum before any lab box. Owned modules are sorted by `(category_rank, tier, cubes_to_unlock, id)` — general → offensive → defensive → other.
2. **Unlock gate.** After each module attempt, the orchestrator refreshes state, runs `check_unlock_gate`, and only opens the next module when all questions are answered AND the cube balance has updated.
3. **Labs after academy.** Once the curriculum is exhausted (or the operator decides), the top-level training script pivots to real `HTBEnv` rollouts (Phase 4). Phase 5b owns the academy half; the labs half is the next-up work.

### Verification (delivered)
- `pytest tests/academy/`: **127+ tests** covering: dataclasses, mock session lifecycle, ranked-candidate answerer for every question type + cheat-sheet matcher + junk filter, curriculum eligibility / ordering / unlock gate, orchestrator end-to-end (study-only + auto-submit + multi-module gate enforcement), auto-demo-writer turn shape (intro + cheat + section_read + answer + sandbox_cmd), MITRE mapping, demo coverage CLI, and wizard safety guards (lab-flag detection, auto-submit refusal).
- Real-academy walks completed for every module the research account owns:
  - **Module 9** "Learning Process" — 20 sections, 0 questions, 21 turns, 75.6 KB theory.
  - **Module 15** "Intro to Academy" — 8 sections, 0 questions, 10 turns (intro + cheat + sections), 19.4 KB.
  - **Module 18** "Linux Fundamentals" — 30 sections, 26 questions, 58 turns, 256.7 KB. Cheat sheet: 72 rows. ATT&CK: T1018, T1057, T1059.004, T1083.
  - **Module 87** "Setting Up" — 22 sections, 2 questions, 25 turns, 107.8 KB.
- `python scripts/htb_academy_list_modules.py` shows every module with state + cost.
- `python scripts/academy_coverage.py` shows the demo set: 5 demos, 90+ section_read turns, 28 answer turns, 4 distinct ATT&CK techniques tagged.
- `python scripts/htb_academy_walk_all.py` walks every owned module with gate enforcement.

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

**Goal:** Rigorous, automated evaluation organized by MITRE ATT&CK matrix and tactic. No more "looks good in tensorboard" judgments.

### Eval suite
- **Enterprise:** 15 held-out boxes (10 easy, 5 medium) — *never seen during training, BC, or RM labeling*.
- **Mobile (v0.2):** 8 held-out APKs from publicly-available crackme + intentionally-vulnerable apps (e.g., DIVA, InsecureBankv2).
- **ICS (v0.3):** 5 held-out OpenPLC+ConPot scenarios with synthetic process logic.
- Per target, 5 evaluation episodes (different starting seeds/timeouts).

### Metrics
- **Outcome metrics:**
  - **Foothold rate** — % of episodes where the agent achieves user-level access (Enterprise: shell; Mobile: app code execution; ICS: PLC config read).
  - **Root rate** — % of episodes with privileged access (Enterprise: root/SYSTEM; Mobile: device admin; ICS: PLC write/control).
  - **Flag/objective time** — median wall-clock to capture or completion (failures count as cap+1).
  - **Command efficiency** — flags-per-action.
- **ATT&CK coverage metrics (per matrix):**
  - **Technique attempt coverage** — % of registered techniques the agent has executed *successfully* at least once across all eval episodes.
  - **Tactic completion rate** — for each of the 14/14/12 tactics, fraction of episodes in which the agent completed it.
  - **Killchain depth** — distribution of max sequential tactics chained per episode (median + p90).
  - **Cross-tactic transfer** — train on tactics A,B,C, hold out tactic D, measure D performance after fine-tune.
- **Health metrics:**
  - **KL-from-BC** — how far the policy drifted from the BC reference.
  - **Vocab coverage** — % of tools used at least once (low coverage → policy collapse).
- Regression tests: gate any merge to `main` on no metric regressing > 5 % vs the previous tagged release.

### Critical files
- `src/htbrl/eval/harness.py` — eval runner.
- `src/htbrl/eval/metrics.py` — metric definitions.
- `src/htbrl/eval/attack_metrics.py` — ATT&CK-specific aggregations.
- `scripts/eval.py` — CLI: `python scripts/eval.py --checkpoint ckpt/v3.pt --suite eval_v1.yaml`.
- `scripts/coverage_report.py` — pretty-prints per-tactic technique coverage.
- `tests/regression/` — pinned eval baselines.

### Verification
Manual: run eval on Phase 9 checkpoint, confirm metrics are computed and CSV written, ATT&CK heatmap rendered (one cell per technique × success rate). Then on Phase 10 checkpoint, confirm regression test gate works.

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

# ===== v0.2 — Mobile matrix (weeks 17–22) =====

## Phase 13 — Mobile attacker setup + tool registry (Weeks 17–18)

**Goal:** Set up the Android attack lab and a Mobile-matrix tool registry.

### Lab
- **Android emulator:** Genymotion or Android Studio AVD with x86_64 Android 11/12 image. Runs on Windows host (uses CPU virtualization — no GPU pressure). One emulator per parallel env, capped at 4 concurrent (2 GB RAM each).
- **Tool stack on Kali:** ADB (already there), Frida + frida-server, MobSF (running as a Docker container — analysis API), drozer, jadx, apktool, Objection, AndroBugs.
- **Sample target apps:** vulnerable Android crackmes from public sources (DIVA, InsecureBankv2, OWASP Goatdroid, vuldroid). All open-source, intentionally vulnerable, no IP issues.

### Mobile tool registry (`registry/mobile_*.yaml`)
- ~80 tools across 14 Mobile-matrix tactics. Examples:
  - **Initial Access:** `apk_install_via_adb`, `frida_inject`, `mobsf_static_analysis`.
  - **Discovery:** `adb_list_packages`, `objection_classes`, `drozer_attack_surface`.
  - **Credential Access:** `apk_extract_strings`, `frida_dump_keychain`, `objection_keychain_dump`.
  - **Collection:** `adb_pull_app_data`, `mobsf_dynamic_logs`.
  - **Exfiltration:** `adb_pull_database`, `frida_intercept_https`.
- Each tool tagged `matrices: [mobile]` (a few — like `nmap` of the emulator's IP — are `[enterprise, mobile]`).

### Critical files
- `src/htbrl/env/mobile_env.py` — Mobile Gymnasium env (ADB-driven, not SSH).
- `src/htbrl/env/adb_session.py` — ADB session wrapper analogous to ssh_session.
- `src/htbrl/env/parsers/mobsf.py`, `parsers/objection.py`, `parsers/drozer.py`.
- `src/htbrl/tools/registry/mobile_*.yaml`.

### Verification
Hand-scripted policy installs DIVA, runs MobSF static analysis, parses output. Episode reward > 0; ADB sessions release cleanly. `pytest tests/mobile/` covers the new tools.

---

## Phase 14 — Mobile model expansion + cross-matrix training (Week 19)

**Goal:** Extend the policy to handle the Mobile action space without forgetting Enterprise.

### Approach
- Freeze the trunk + Enterprise tool embeddings. Add Mobile tool embedding rows + Mobile matrix-embedding bias.
- Fine-tune on a 70/30 mix of Enterprise/Mobile rollouts. Tight KL penalty against the Enterprise-trained reference policy keeps the trunk stable.
- New PPO config: `configs/train/ppo_mobile.yaml` with mobile-specific reward annealing (Mobile flags are subtler — successful data-exfil from a sandboxed app is the rough analog of a root flag).

### Verification
- Mobile foothold rate (app instrumented or static analysis succeeded) ≥ 50 % on held-out Mobile suite.
- **No regression** on Enterprise eval suite (≥ 95 % of pre-expansion metrics preserved).

---

## Phase 15 — Mobile RM + RLHF (Weeks 20–22)

**Goal:** Mobile-specific human feedback (different "good behavior" than Enterprise — e.g., Frida hooking that crashes the app is bad even if it leaks data).

- Reuse the FastAPI feedback UI; add Mobile rendering pages (decompiled snippets, MobSF reports, ADB logs).
- Train a Mobile reward model (4-layer transformer like Enterprise RM) on ~600 Mobile preference comparisons.
- Composite reward: `r_env + α·r_rm_mobile + β·r_rnd − c·KL`. Same structure, separate RM per matrix.

### Verification
Held-out preference accuracy ≥ 65 % on the Mobile RM. RLHF Mobile agent beats non-RLHF Mobile agent on human-eval blind comparison.

---

# ===== v0.3 — ICS matrix (weeks 23–28) =====

## Phase 16 — ICS lab + tool registry (Weeks 23–24)

**Goal:** Stand up an OT/ICS lab the agent can safely poke at, and a tool registry for the 12 ICS tactics.

### Lab
- **OpenPLC** running a synthetic ladder-logic process (e.g., a tank-fill PLC program from public examples). Modbus TCP exposed.
- **ConPot** as a SCADA / Siemens S7 honeypot — gives the agent a Siemens-flavored target without buying real hardware.
- **GRFICSv2** (the Graphical Realism Framework for ICS Security) for a more realistic multi-PLC water-treatment scenario. Runs in VirtualBox on the Windows host.
- All ICS targets on a dedicated isolated `192.168.95.0/24` network. **The IP allowlist makes it physically impossible for the agent to touch a real ICS device** — the env refuses any action against an IP outside this CIDR.

### ICS tool registry (`registry/ics_*.yaml`)
- ~60 tools across 12 ICS tactics. Examples:
  - **Discovery:** `plcscan`, `nmap_modbus_nse`, `s7scan`, `enip_scan`.
  - **Initial Access:** `modbus_login_default`, `s7_default_creds`.
  - **Persistence / Impair Process Control:** `modbus_write_coil`, `s7_upload_block` (only allowed against the lab CIDR, with extra confirmation flag in config).
  - **Inhibit Response Function:** `modbus_force_listen_only_mode`.
  - **Collection:** `modbus_read_holding_registers`, `s7_data_block_read`.
- Tools tagged `matrices: [ics]`. Some (network probes) shared with Enterprise.

### Critical files
- `src/htbrl/env/ics_env.py` — ICS Gymnasium env. Runs against the lab CIDR, refuses everything else.
- `src/htbrl/env/parsers/modbus.py`, `parsers/s7.py`.
- `src/htbrl/tools/registry/ics_*.yaml`.

### Safety
- `configs/env/ics.yaml` requires an explicit `allowlist_cidr: 192.168.95.0/24` and an `i_understand_this_can_break_real_industrial_systems: true` flag. Loader refuses to start the ICS env without both. This is overkill for a local lab but I want it impossible to misuse the registry against unintended targets.

### Verification
Hand-scripted policy: nmap → plcscan → modbus_read_holding_registers → modbus_write_coil. Tank-level register changes; OpenPLC log confirms write. No traffic leaves the lab CIDR (verified via `tshark` capture on the host).

---

## Phase 17 — ICS model expansion (Week 25)

Same recipe as Phase 14 but for ICS: add ICS tool embeddings + matrix bias, freeze trunk, fine-tune on 60/20/20 Enterprise/Mobile/ICS rollouts. Verify no regression on prior matrices.

---

## Phase 18 — ICS RM + cross-matrix RLHF + final eval (Weeks 26–28)

- Train an ICS reward model on ~400 preferences (smaller because the action space is smaller; "did the agent achieve process-control disruption *without* causing a real-world-equivalent safety event" is the central judgment call).
- Final eval suite runs all three matrices end-to-end. Produces a single ATT&CK heatmap PDF + JSON with per-matrix per-tactic per-technique success rates. This is the deliverable.

---

## Critical files reference (final)

These are the files most likely to need iteration; biased toward the algorithmic core:

- `src/htbrl/model/transformer.py`, `model/policy.py`, `model/heads.py`
- `src/htbrl/algo/ppo.py`, `algo/gae.py`, `algo/rnd.py`, `algo/kl_ctrl.py`, `algo/rlhf.py`
- `src/htbrl/env/htb_env.py`, `env/mobile_env.py`, `env/ics_env.py`, `env/ssh_session.py`, `env/adb_session.py`, `env/rewards.py`
- `src/htbrl/tools/registry/*.yaml`, `tools/loader.py`, `tools/coverage.py`
- `src/htbrl/tokenizer/bpe.py`
- `src/htbrl/rm/model.py`
- `src/htbrl/feedback/server.py`
- `src/htbrl/eval/attack_metrics.py`
- `scripts/train_bc.py`, `train_ppo.py`, `train_rm.py`, `eval.py`, `coverage_report.py`

---

## Verification (end-to-end)

### v0.1 (week 16) — Enterprise gate
1. `python scripts/smoke_test.py` exits 0 (Phase 0 sanity).
2. `pytest -q` passes (per-phase unit + integration tests).
3. `python scripts/eval.py --checkpoint ckpt/final.pt --suite eval_v1.yaml` shows on the held-out Enterprise boxes:
   - **Foothold rate ≥ 60 %**, **user-flag rate ≥ 30 %**.
   - **ATT&CK technique attempt coverage ≥ 40 %** of registered Enterprise techniques.
   - **Tactic completion** ≥ 1 in at least 8 of 14 Enterprise tactics across the eval suite.
   - **Median killchain depth ≥ 3** tactics.
4. The feedback UI at `localhost:8765` lets you label new pairs, retrain the RM in ≤ 30 min on the 3060, and the next PPO iteration uses the new RM.
5. Peak VRAM during full training stays under 11 GB; CPU saturates around 70–80 % during rollout.
6. No pretrained model or external LLM was used at any step — `git log -p` over the codebase has zero downloads of model weights or pretrained tokenizer files.

### v0.2 (week 22) — Mobile gate
- Mobile foothold rate ≥ 50 %; **no Enterprise regression** (each Enterprise metric within 5 % of v0.1 baseline).
- ATT&CK technique attempt coverage ≥ 30 % across Mobile-matrix techniques.

### v0.3 (week 28) — ICS gate (final)
- ICS objective rate ≥ 50 % (process-control read or controlled write achieved).
- **No regression on Enterprise or Mobile** (within 5 % of prior baselines).
- Final ATT&CK heatmap PDF generated covering all three matrices.

---

## Risk register (so we don't fool ourselves)

- **HTB doesn't allow snapshot/reset.** Mitigation: develop on local Vulnhub/Metasploitable + custom Docker lab; HTB is for *evaluation*, not training reset loops. Phase 4 already accounts for this.
- **Demonstration data is the bottleneck.** If you can't sustain 800–1500 trajectories per matrix, BC will be weak and PPO will struggle. Mitigation: start collecting demos *during* Phase 1–2, not waiting for Phase 5. For Mobile/ICS, demonstration coverage is the rate-limiter on each track's start.
- **Reward hacking the RM.** Mitigation: KL hard cap + scheduled RM refresh + RM-min ensemble in Phase 9. ATT&CK coverage bonus also vulnerable to gaming (agent runs every tool once for the bonus); mitigated by aggressive annealing schedule.
- **3060 thermal throttling on long runs.** Mitigation: target ≤ 80 °C; cap `power_limit` via `nvidia-smi -pl` if needed; checkpoint every 100 k steps so a thermal trip costs < 30 min.
- **Action vocabulary is too narrow.** Mitigation: schema is hot-loadable; new tools can be added between training runs without retraining the whole policy (only the tool-embedding row for new tools is fresh, rest is reused — described in Phase 1's embedding scheme).
- **Cross-matrix catastrophic forgetting.** Adding Mobile/ICS could degrade Enterprise performance. Mitigation: freeze trunk during expansion (Phases 14, 17), use mixed-matrix rollouts with ratios tuned per phase, gate releases on no-regression test.
- **ICS lab safety.** A misconfigured allowlist or a pivot through a host with two NICs could let the agent reach a real industrial network. Mitigation: physically isolated VirtualBox network for ICS lab; double-flag config (allowlist CIDR *and* the explicit `i_understand_this_can_break_real_industrial_systems: true`); the env literally refuses to start without both.
- **Mobile emulator ↔ GPU contention.** Genymotion uses CPU virtualization, so emulator + PyTorch don't fight for VRAM, but they *do* fight for RAM. Mitigation: cap concurrent Mobile envs at 4 (8 GB total emulator RAM) so the rollout buffer + transformer activations still fit in remaining 8 GB.
- **Timeline scope creep.** 28 weeks is aggressive across three matrices. Mitigation: each track has a hard release gate; if v0.1 (Enterprise) slips past week 16 by > 4 weeks, defer Mobile/ICS to a v1.x rather than compress quality.
