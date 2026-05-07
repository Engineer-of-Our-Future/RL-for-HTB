# RL for HTB

Pure-PyTorch reinforcement-learning agent that learns to operate in HackTheBox-style attacker environments by interacting with target VMs and incorporating human feedback. **Built from absolute ground up — no pretrained models, no LLM dependencies, no third-party AI weights.**

The full 16-week, 12-phase roadmap lives in [`PLAN.md`](PLAN.md).

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
**Phases 0–9 + 11 + 5b auto-learner.** Tool registry covers all 14 Enterprise tactics (73 tools) + 7 ICS dual-tagged. Full PPO loop runs end-to-end against the stub env; real `HTBEnv` works against a WSL Kali attacker. HTB Academy auto-learner with study-only / auto-submit modes ships under `htbrl.academy`. **292 tests passing.**

## HTB Academy auto-learner (Phase 5b)
A separate progression path that reads HTB Academy modules, optionally drives the per-module SSH sandbox to derive answers, and writes every interaction to the same `Demonstration` format the BC trainer reads. Two modes:

- **`study_only` (default).** Reads modules + practices in the sandbox, **never POSTs an answer**. Use this for safe data collection.
- **`auto_submit`.** Actually submits answers to the academy. Requires `--enable-auto-submit --i-accept-academy-tos-risk`. **Auto-completing academy modules to farm cubes/XP is a gray-zone use of an educational platform and may violate HTB's Terms of Service. Use only on a research account you accept might be banned.** I do not recommend this mode.

The transport layer is abstract (`AcademySession` ABC). The bundled `MockAcademySession` lets you exercise the full pipeline end-to-end without touching the real site:

```powershell
python scripts\htb_academy.py --transport mock --max-modules 1
# writes data\auto_demos\academy_m1.msgpack.gz which BC training can mix in
```

A real Playwright-based `PlaywrightAcademySession` is intentionally NOT included by default — implementing it pulls in browser binaries, and the sober choice is to leave it as a user-supplied extension.

## Quick install (Windows host)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts\smoke_test.py
```

Without WSL/Kali set up, the SSH check is skipped and only the GPU portion runs.

## WSL2 Kali attacker (Phase 4)
The real env needs an SSH-reachable Kali host. The repo's `HTBEnv` opens a persistent SSH session into Kali and renders structured tool calls into bash via the registry. Set up Kali like so (one-time, ~5 min, ~700 MB download):

```powershell
# 1. install the WSL2 distro (no first-launch interactive prompt)
wsl --install -d kali-linux --no-launch

# 2. inside Kali, set up sshd + htbrl user with sudo + your SSH key
$pubkey = Get-Content ~\.ssh\htbrl_kali.pub
wsl -d kali-linux --user root -- bash -c "
  apt-get update -qq && apt-get install -y -qq openssh-server sudo nmap smbclient curl dnsutils
  useradd -m -s /bin/bash -G sudo htbrl 2>/dev/null || true
  echo 'htbrl ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/99-htbrl
  mkdir -p /home/htbrl/.ssh && echo '$pubkey' > /home/htbrl/.ssh/authorized_keys
  chmod 700 /home/htbrl/.ssh && chmod 600 /home/htbrl/.ssh/authorized_keys
  chown -R htbrl:htbrl /home/htbrl/.ssh
  sed -i 's/#\?Port 22.*/Port 2222/' /etc/ssh/sshd_config
  sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
  ssh-keygen -A
  printf '[boot]\ncommand = service ssh start\n[user]\ndefault = htbrl\n' > /etc/wsl.conf
  service ssh start
"

# 3. test from PowerShell
$env:HTBRL_KALI_HOST = 'htbrl@127.0.0.1:2222'
$env:HTBRL_KALI_KEY  = "$HOME\.ssh\htbrl_kali"
python scripts\smoke_test.py    # both GPU and SSH should be OK
```

If you don't already have a key, generate one first: `ssh-keygen -t ed25519 -f ~/.ssh/htbrl_kali -N ""`.

The `[boot] command = service ssh start` in `/etc/wsl.conf` makes sshd auto-start on every WSL boot — but WSL has to be **fully shut down** (`wsl --shutdown`) to re-read the config, not just have the distro stop on idle.

To run pytest including the SSH-and-Kali tests:

```powershell
$env:HTBRL_KALI_HOST = 'htbrl@127.0.0.1:2222'
$env:HTBRL_KALI_KEY  = "$HOME\.ssh\htbrl_kali"
pytest -q
```

## Project layout
```
RL-for-HTB/
├── PLAN.md                   # full 16-week roadmap (read this first)
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
Proprietary. Personal-use project — see `pyproject.toml`.
