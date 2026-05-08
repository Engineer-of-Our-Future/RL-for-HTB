# Academy walking — Phase 5b auto-learner

The `htbrl.academy` package walks HTB Academy modules, attempts the
per-section questions, and writes one `Demonstration` per module to
`data/auto_demos/` for BC training to mix in.

## Modes

| Mode | What it does | Default? | When to use |
|---|---|---|---|
| `study_only` | Reads content, runs sandbox, **never POSTs answers**. | ✅ Yes | Safe data collection on any account. |
| `auto_submit` | Wizard fills the academy input + clicks Submit when top candidate's confidence ≥ `--auto-confidence`. Lab-flag-shaped questions are NEVER auto-submitted. | ❌ Opt-in | Only on a research account you accept might be banned per HTB ToS. |

**Auto-completing academy modules to farm cubes/XP is gray-zone use of
an educational platform and may violate HTB's Terms of Service.** The
default is `study_only`. The submit path exists for users who
explicitly accept the risk on a research account.

## How the walker reaches the academy

1. The operator launches Chrome with CDP debugging:
   ```powershell
   .\scripts\start_chrome_for_htb.ps1
   ```
2. The operator logs into HTB Academy manually in that Chrome window.
   This is on purpose — Cloudflare's bot detection blocks Playwright's
   launch markers, so we attach to a *user-driven* Chrome instead.
3. The walker attaches via the Chrome DevTools Protocol on port 9222
   (`htbrl.academy.cdp_walker.open_cdp`). All academy DOM reads + form
   fills happen through CDP.

## CLI entry points

```
scripts/htb_academy_login_check.py     # multi-mode login probe
scripts/htb_academy_run.py             # single-module study-only walker
scripts/htb_academy_wizard.py          # operator-in-the-loop walker
scripts/htb_academy_walk_all.py        # multi-module driver, gate-checked
scripts/htb_academy_list_modules.py    # discovery (table of owned/in_progress)
scripts/academy_coverage.py            # per-demo coverage report
```

`htb_academy_wizard.py` is the most useful in practice — it shows you
the model's top-N candidates per question and lets you accept,
override, or skip.

## Per-question probe pipeline (wizard)

The wizard tries each probe in order; the first one that returns a
non-None tuple wins.

| # | Probe | When it fires |
|---|---|---|
| 1 | `target_runner.probe_target_for_answer` | HTTP shape: server header, JSON field, login + search, CRUD chain, HTML endpoint discovery |
| 2 | `lfi_runner.probe_via_lfi` | Filter-bypass / "read /flag.txt" / "LFI" prompts |
| 3 | `ssh_runner.probe_via_ssh` | "SSH to X with user U password P" + a shell-shape question (kernel, inode, file count, etc.) |
| 4 | `rfi_runner.probe_via_rfi` | RFI prompts AND `--kali-host` set (lazy listener spins up via SSH-attached Kali) |
| 5 | `target_runner.probe_code_blocks_for_flag` | Replays cURL examples from the section's code blocks against the spawned target |
| 6 | Heuristic answerer's top candidate | Always (fallback) |

The matched probe's `(answer, rationale, confidence)` is inserted as
candidate-0. Operator still has the final say on Submit.

## Wizard usage examples

### Plain (no RFI, no auto-submit)

```powershell
python scripts\htb_academy_wizard.py --module-id 18
```

### With auto-submit

```powershell
python scripts\htb_academy_wizard.py `
  --module-id 18 `
  --auto-submit --auto-confidence 0.85 `
  --i-accept-academy-tos-risk
```

Lab flag questions stay manual regardless of `--auto-submit`.

### With RFI listener armed (mod 23 RFI sections)

```powershell
$env:HTBRL_KALI_HOST     = 'claude@192.168.1.219:22'
$env:HTBRL_KALI_PASSWORD = 'claude'

python scripts\htb_academy_wizard.py `
  --module-id 23 `
  --kali-listen-ip 10.10.15.54   # Kali tun0 IP (NOT the SSH IP!)
```

The listener is opened lazily on the first RFI-shaped question and
torn down in the wizard's `finally` clause.

## Cube budget gate

The academy charges cubes to *open* a module. Opening a second one
before the first is finished wastes the operator's account budget.

`htbrl.academy.curriculum.check_unlock_gate(current_module,
answered_qids, cubes_before, cubes_after)` enforces:

> open new module only if all questions are answered AND cube balance
> is updated.

Both halves are independently toggleable. The orchestrator runs the
gate before each new module open; the wizard runs it once at end of
walk and prints a verdict.

## Demo turn ordering

Per module, the auto-demo writer emits BC-friendly turns in this
order:

```
academy_module_intro       (prelude + takeaways + conclusion)
academy_cheat_sheet        (canonical command/desc table)
academy_section_read        (theory text)        ┐
academy_answer * N          (question attempts)  ├─ per section
academy_sandbox_cmd         (when sandbox used)  ┘
... repeated per section ...
```

Synthetic tool names (`academy_module_intro`, `academy_cheat_sheet`,
`academy_section_read`, `academy_answer`, `academy_sandbox_cmd`) keep
academy turns distinguishable from real-env demos in the BC trainer.

Every turn carries `techniques_attempted` / `techniques_succeeded`
from `htbrl.academy.mitre_mapping`.

## When the walker is enough vs not

- **Walker is enough:** factual questions ("what server header", "the
  inode of /etc/shadow.bak") — the probe pipeline answers most of
  these in one round-trip.
- **Walker is not enough:** multi-step CTF questions (chattr Skills
  Assessment, sumace Skills Assessment) where the answer requires
  chained registration → SQLi → flag extraction. These are operator
  work for now; they're explicitly out of scope per `LABS_PLAN.md`.

## Modules walked so far

```
data/auto_demos/
├── academy_module_9.msgpack.gz
├── academy_module_15.msgpack.gz
├── academy_module_18.msgpack.gz
├── academy_module_18_wizard.msgpack.gz
├── academy_module_23.msgpack.gz
├── academy_module_33.msgpack.gz
├── academy_module_34.msgpack.gz
├── academy_module_35.msgpack.gz
├── academy_module_35_wizard.msgpack.gz
├── academy_module_49.msgpack.gz
├── academy_module_74.msgpack.gz
└── ...
```

To add another, just run the wizard against it:
`python scripts\htb_academy_wizard.py --module-id <N>`.
