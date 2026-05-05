# RL for HTB

Pure-PyTorch reinforcement-learning agent that learns to operate in HackTheBox-style attacker environments by interacting with target VMs and incorporating human feedback. **Built from absolute ground up — no pretrained models, no LLM dependencies, no third-party AI weights.**

The full 16-week, 12-phase roadmap lives in [`PLAN.md`](PLAN.md).

A more precise plan before the first release:
The plan is broken down into 3.7 months => 16 weeks => 112 days, and 12 stages.

## Hardware target
This codebase is tuned specifically for:

- **GPU:** NVIDIA RTX 3060, 12 GB VRAM (Ampere)
- **RAM:** 16 GB system memory
- **CPU:** AMD Ryzen 7 5700G (8 cores / 16 logical processors)

If you run on different hardware, expect to revisit the model size, batch size, and number of parallel rollout workers in `configs/`.

## High-level architecture
- **Agent + PyTorch** runs on the Windows host (uses the 3060 natively, no virtualization overhead).
- **Kali attacker** runs in WSL2 (or a separate VM). Hosts pentesting tools and the OpenVPN connection to HTB.
- **Communication:** the env wrapper opens a persistent SSH session into Kali, executes structured tool calls, and parses the output into structured + tokenized observations.

```
    Windows host                                WSL2 / Kali                     HTB
+---------------+    SSH    +-----------------+    OpenVPN    +-----------------+
|  PyTorch +    |<--------->|  tool registry  |<------------->|  target VM(s)   |
|  RL trainer   |           |  (nmap, ...)    |               |  (10.10.x.x)    |
+---------------+           +-----------------+               +-----------------+
```

## Status
**Phase 0 — project bootstrap.** Repo skeleton in place; smoke test wires up `torch` and SSH-to-Kali. No model code, no training yet.

## Quick install (Windows host)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts\smoke_test.py
```

The smoke test prints your GPU name + VRAM and (optionally) `whoami` over SSH to Kali. Set the SSH endpoint via env var:

```powershell
$env:HTBRL_KALI_HOST = "kali@127.0.0.1:2222"
python scripts\smoke_test.py
```

## Project layout
```
RL-for-HTB/
├── pyproject.toml
├── README.md (this file)
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
├── scripts/                  # train_bc.py, train_ppo.py, eval.py, smoke_test.py, ...
├── tests/                    # unit + integration + regression
└── docs/
```

## Allowed dependencies
The "no LLMs / no pretrained models" rule is enforced at the dependency level:

- **Allowed:** `torch`, `numpy`, `gymnasium`, `pexpect`, `paramiko`, `pyyaml`, `hydra-core`, `tensorboard`, `fastapi`, `pydantic`, `bitsandbytes` (kernels only), `tqdm`, `msgpack`, `regex`.
- **Forbidden:** `transformers` (HuggingFace), `tokenizers`, `trl`, `peft`, `accelerate` (when used to load weights), or any package whose primary purpose is to load pretrained model weights.

If you need to add a dependency, ask: *does this ship pretrained weights or use them at install time?* If yes, find another way.

## Safety
The env wrapper enforces a **target-IP allowlist** in every config. The agent will refuse to launch a tool against any IP outside the configured CIDR. This prevents an exploration-driven policy from accidentally scanning the public internet during rollouts. Always double-check `configs/env/<your-config>.yaml` before running training.

## License
Proprietary. Personal-use project — see `pyproject.toml`. If you want to use this project as `Personal-use`, then just replace the `name` in the `pyproject.toml` file in the line `authors = [{ name = "Hmm" }]`
