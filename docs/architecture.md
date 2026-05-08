# Architecture

End-to-end picture of the RL-for-HTB agent. Sister docs:

- `docs/training_pipeline.md` — BC → PPO → RM → RLHF → eval, command by command.
- `docs/academy_walking.md` — running the Phase 5b auto-learner.
- `docs/labs_walking.md` — moving from academy to live HTB lab boxes.
- `docs/troubleshooting.md` — VPN / SSH / allowlist failures, what to check.
- `LABS_PLAN.md` — the post-academy execution plan.
- `PLAN.md` — the original 16-week phase-by-phase build.

## Where things run

```
+========================== Windows host (3060 + 16 GB RAM) ==========================+
|                                                                                      |
|   PyTorch:  policy + critic + reward model + PPO trainer + replay buffer + tokenizer |
|   CDP:      browser-attached academy walker (port 9222)                              |
|   FastAPI:  preference labelling UI (localhost:8765)                                 |
|   Tests:    pytest (530 collected; 7 skipped without HTBRL_KALI_HOST)                |
|                                                                                      |
+-----------------------|----------------------------|---------------------------------+
                        | SSH (paramiko, password    |
                        | OR key auth via env vars)  |
                        v                            v
+========= Kali attacker (laptop OR WSL2) ==========++===== Browser (academy CDP) ===+
|                                                  ||                                |
|  - openvpn (academy + HTB lab tunnels)           ||  Chrome --remote-debugging-    |
|  - nmap / gobuster / hydra / smbclient           ||  port=9222 attaches the wizard |
|  - python3 (RFI listener via http.server)        ||  to academy.hackthebox.com     |
|  - persistent shell via SSHSession.invoke_shell  ||  (no Playwright launch -> no   |
|                                                  ||   Cloudflare bot detection)    |
+----------|--------------------|------------------++--------------------------------+
           | OpenVPN tun0       | OpenVPN tun0 (academy)
           v                    v
+- HTB labs ---------+  +- HTB Academy targets -+
| 10.10.10.0/24      |  | 154.57.x.x (external) |
| 10.129.x.x         |  | 10.129.x.x (internal) |
+--------------------+  +-----------------------+
```

## Why Windows host + Kali attacker (not Kali everywhere)

PLAN.md decided this fork up-front:

- **GPU lives on Windows.** RTX 3060 is on the host; running the agent
  inside WSL2 or a VM costs 10-20% CUDA throughput and complicates
  driver upgrades.
- **Pentest tooling lives on Kali.** `nmap`, `gobuster`, `hydra`, etc.
  are easier to keep current on Kali. The agent renders a structured
  `Action(tool_id, slots)` into a bash command and ships it through
  the `SSHSession`; output is parsed back into structured features
  by `htbrl.env.parsers.*`.
- **OpenVPN lives on Kali.** HTB's lab and academy VPNs both run as
  OpenVPN on the attacker. Putting the tunnel on Kali keeps the
  Windows host's network untouched and avoids leaking the agent's
  rollout traffic into the operator's normal browsing.

Trade-off: a hung Kali wedges the rollout. The env wrapper caps each
command with a per-step timeout (default 120s) and fails the episode
gracefully when the SSH transport drops.

## Module layout

```
src/htbrl/
├── tokenizer/         # BPE, special-token registry  (Phase 2)
├── tools/             # YAML registry + slot validation  (Phase 1)
├── model/             # transformer, heads, init  (Phase 3)
├── env/               # HTBEnv + SSHSession + parsers  (Phase 4)
├── data/              # demo / rollout / preference datasets  (Phases 5/6/8)
├── algo/              # PPO, GAE, RND, KL controller, RLHF composite reward  (Phases 6/7/9)
├── rm/                # reward model architecture  (Phase 8)
├── feedback/          # FastAPI labelling UI + storage  (Phase 8)
├── eval/              # eval harness + metrics  (Phase 11)
├── utils/             # checkpoint registry, optim helpers  (Phase 10/12)
└── academy/           # Phase 5b auto-learner — runs *before* labs
    ├── session.py            # transport ABC + MockAcademySession
    ├── cdp_walker.py         # Chrome DevTools Protocol attach + section walker
    ├── target_runner.py      # HTTP probe (paramiko-free, urllib only)
    ├── ssh_runner.py         # SSH probe (paramiko, used by mod 18 / 33 questions)
    ├── lfi_runner.py         # LFI bypass payload generator (mod 23 sec 1491+)
    ├── rfi_runner.py         # RFI listener via SSH-attached Kali (mod 23 sec 254 etc)
    ├── answerer.py           # ranked candidate proposer
    ├── orchestrator.py       # multi-module driver with unlock gate
    ├── auto_demo_writer.py   # converts academy turns to BC-friendly demos
    ├── curriculum.py         # module ordering + cube-budget gate
    ├── mitre_mapping.py      # ATT&CK technique tags per module
    └── walkthrough.py        # markdown writer for human review
```

## The wizard's per-question probe pipeline

When the academy wizard reaches a question, it consults probes in this order:

1. `target_runner.probe_target_for_answer` — HTTP shape (server header,
   JSON field, login + search, CRUD chain, HTML endpoint discovery).
2. `lfi_runner.probe_via_lfi` — LFI bypass walker (php://filter base64 →
   recursive `....//` → URL-encoded → approved-prefix variants).
3. `ssh_runner.probe_via_ssh` — 9 shell-shape patterns (kernel version,
   inode lookup, .ext file count, etc.). Creds parsed from prompt.
4. `rfi_runner.probe_via_rfi` — only when the operator passed `--kali-host`
   (and `--kali-listen-ip`); listener spins up on first RFI prompt.
5. `target_runner.probe_code_blocks_for_flag` — replay any cURL example
   in the section's code blocks against the spawned target.
6. Heuristic answerer's top candidate.

The first probe that returns a non-None tuple wins; its `(answer,
rationale, confidence)` is inserted as candidate-0 so it ranks above
all heuristic candidates.

## The training pipeline

```
                +---------------+    +---------------+
data/auto_demos | academy turns |    | manual lab    | data/demos
data/demos      | (5b auto-     |    | walks (5)     |
                | learner)      |    |               |
                +-------+-------+    +-------+-------+
                        |                    |
                        +--------+-----------+
                                 |
                                 v
                       +----------------------+
                       | scripts/train_bc.py  |   Phase 5
                       +----------+-----------+
                                  |
                                  v
                  +------------------------------+
                  | checkpoints/bc-...-vN.pt     |
                  +------------+-----------------+
                               |
                +--------------+-------------+
                |                            |
                v                            v
   +-----------------------+     +------------------------+
   | scripts/train_ppo.py  |     | scripts/serve_feedback |   Phase 8
   |  --env-type htb       |     |  (operator labels prefs)
   |  --ref-checkpoint X.pt|     +------+-----------------+
   +-----------+-----------+            |
               |                        v
               |             +-----------------------+
               |             | scripts/train_rm.py   |   Phase 8
               |             +------+----------------+
               |                    |
               +-------+------------+
                       |
                       v
            +-------------------------+
            | scripts/train_ppo.py    |   Phase 9 (RLHF)
            |  --rm-checkpoint Y.pt   |
            |  --ref-checkpoint X.pt  |
            +------------+------------+
                         |
                         v
            +-------------------------+
            | scripts/eval.py         |   Phase 11
            |  --suite holdout_v1     |
            +-------------------------+
```

Each box is a real script that smoke-runs. Production-scale runs
take GPU hours / human labelling time; see `LABS_PLAN.md` step-by-step.

## Data flow per env step

```
agent picks (tool_id, slots)
        |
        v
ActionVocabulary renders the bash command via the YAML template
        |
        v
HTBEnv allowlist gate (refuses if any IP in the rendered command
                       falls outside the configured CIDRs)
        |
        v
SSHSession.run(command, timeout=...)  -> (stdout, exit_code, timed_out)
        |
        v
parser registry (per-tool: nmap, gobuster, hydra, smbclient, ...)
                       extracts structured features (open_ports,
                       discovered_paths, found_creds, ...)
        |
        v
htbrl.env.rewards: emit shaped reward (+0.1 new port, +0.2 new service,
                                       +0.5 user shell, +1.0 user flag,
                                       +1.5 root shell, +2.0 root flag,
                                       -0.01 step, -0.1 timeout)
        |
        v
encode_state.py: turn-structured token sequence prepended to rolling
                 window (last K=8 turns); long history compressed
                 via a small LSTM into a single soft prefix token.
        |
        v
back to agent for next step
```

## Safety guardrails

- **Allowlist** — `HTBEnv` refuses any rendered command whose extracted
  IPs fall outside the configured CIDRs. Episode ends with a reward
  hit so the policy learns not to escape.
- **Rate limit + per-step timeout** — bounded SSH command runtime so a
  hung target can't consume the rollout budget.
- **No lab-flag auto-submit** — the academy wizard never auto-submits
  questions detected as "lab flag" shape (per the project rule;
  account ban risk).
- **Per-call output cap** — `SshTargetRunner` and `SSHSession` both
  cap stdout/stderr at 16 KB so a `cat /dev/zero` can't fill RAM.
- **No keys / passwords on cmdline** — paramiko handles auth via
  kwargs; nothing reaches `ps`.

## Hardware budget reminder

From `PLAN.md`:

| Resource | Budget | What spends it |
|---|---|---|
| VRAM 12 GB | ~10.5 GB usable | Policy (~1 GB) + Critic (shared trunk) + Ref policy (~1 GB) + RM (~0.4 GB) + AdamW state (~2 GB w/ 8-bit) + activations + rollout cache (~4 GB) |
| RAM 16 GB | ~12 GB usable | Replay/rollout buffers (3-4 GB), tokenizer + vocab tables, demo trajectories, env subprocesses |
| CPU 16 logical | 12 workers for env rollout | Parallel SSH-attached envs sample trajectories while GPU does forward passes |

If a phase blows the budget, the order of remediation is: shrink
`d_model`, gradient checkpointing on transformer blocks, drop
`max_seq`, reduce concurrent rollout workers.
