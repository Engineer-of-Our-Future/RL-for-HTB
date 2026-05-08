# Troubleshooting

Common failure modes you'll hit running the agent + the academy
walker. Sorted by symptom.

## Smoke test / pytest

### `[FAIL] torch import: ...`
Your venv doesn't have PyTorch with CUDA. Fix:

```powershell
.venv\Scripts\activate
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

### `[WARN] CUDA not available`
Either no NVIDIA driver or the wrong torch wheel. Run `nvidia-smi` —
if that fails, install the latest GeForce driver. If it succeeds,
reinstall torch with the cu121 wheel above.

### `[SKIP] HTBRL_KALI_HOST not set`
Expected if Kali isn't up yet. To enable:

```powershell
$env:HTBRL_KALI_HOST     = 'claude@192.168.1.219:22'      # or htbrl@127.0.0.1:2222 for WSL
$env:HTBRL_KALI_PASSWORD = 'claude'                       # OR HTBRL_KALI_KEY pointing to identity file
```

### `pytest -m ssh` shows skips even though Kali is up
Either the env vars weren't exported in the same shell that runs
pytest, or `_KALI_PASSWORD or _KALI_KEY` is empty. Confirm:

```powershell
echo $env:HTBRL_KALI_HOST $env:HTBRL_KALI_PASSWORD
```

### `test_ssh_runs_pentest_tool` skipped
nmap isn't installed on Kali. Fix on the Kali host:

```bash
sudo apt-get update
sudo apt-get install -y nmap
```

### `test_htb_env_runs_nmap_against_loopback_via_kali` skipped
Same as above — nmap isn't installed on Kali. Fix as above.

## Academy walker (Phase 5b)

### `[wizard] failed to attach to http://127.0.0.1:9222: ...`
Chrome isn't running with CDP debugging on port 9222.

```powershell
# stop any running Chrome first, then:
.\scripts\start_chrome_for_htb.ps1
# log into HTB Academy in that Chrome window
# THEN run the wizard
```

### `Cloudflare challenge` / "checking your browser" loop
Don't auto-launch Chrome — let the operator log in manually first.
The walker attaches to a *running* Chrome instance via CDP precisely
to bypass Playwright launch detection.

### Auto-submit fills wrong answer / academy rejects
Wizard's heuristic confidence is below `--auto-confidence` (default
0.85). Either:
- Bump `--auto-confidence 0.95` so only very confident answers fire.
- Stay manual: drop `--auto-submit` and pick from candidates yourself.

### `state=accepted detail='polled' inputs_before=N after=N`
Known false-positive: HTB locks the input briefly during processing,
which the result-poll JS reads as "accepted". The wizard verifies
input count drop before crediting the answer; if `before == after`
the answer is treated as NOT accepted (rejected) regardless of the
state string.

### RFI probe fires but listener fails to start
Check `--kali-host` + `--kali-password` + `--kali-listen-ip` are all
set. The listener IP is the address the **academy target** sees when
reaching the **Kali listener** — usually the Kali OpenVPN tun0
address, NOT the SSH IP. Check Kali:

```bash
ip -4 addr show tun0     # use this IP for --kali-listen-ip
```

If the listener URL works in `curl` from Kali (`curl
http://<tun0>:<port>/shell.php`) but not from the academy target,
the lab target can't reach Kali — confirm Kali is on the same VPN as
the lab target.

## Env wrapper (Phase 4)

### `ValueError: allowlist_cidrs is required`
You instantiated `HTBEnv` with no allowlist. Always pass at least
one CIDR (loopback for tests, `10.10.10.0/24` etc. for HTB labs).

### `obs_text` says "allowlist" / `extras["allowlist_violation"]` set
The agent rendered a command targeting an IP outside the configured
CIDRs. Episode ends with negative reward. Fix:
- Update `configs/env/htb.yaml` to include the right CIDR.
- OR check the slot value the agent emitted — sometimes the policy
  is hallucinating an IP.

### SSH session drops mid-episode
- Kali might have gone idle / been swapped out (WSL2 specifically).
  Run `wsl --shutdown && wsl -d kali-linux` to re-establish.
- VPN tunnel dropped. Check `pgrep openvpn` on Kali; restart if
  missing.
- Network interface on the Windows host changed (e.g. you joined a
  different Wi-Fi). The env wrapper opens a fresh `SSHSession` per
  episode, so this self-heals on the next reset.

### `nmap: not installed` / "Do you want to install it? (N/y)"
Kali doesn't have nmap. Fix as above (`sudo apt-get install -y
nmap`). The env tests now pre-flight `command -v nmap` and skip
gracefully when missing — your training run will fail differently
(empty parser output) until nmap is installed.

## Training scripts

### `train_bc.py` says "loaded 0 demos"
Wrong `--demo-root` or no `.msgpack.gz` files there. Verify:

```powershell
ls data\auto_demos\*.msgpack.gz
ls data\demos\*.msgpack.gz
```

### `train_ppo.py` OOMs
The 12 GB 3060 budget is tight. In order:
1. Add `--use-8bit-adamw` (saves ~1.5 GB at full size).
2. Add `--grad-checkpointing` (cuts activation memory ~60%, adds
   ~25% step-time).
3. Reduce `--n-envs` from 8 → 4 → 2.
4. Reduce `--max-seq-len` from 1024 → 768.
5. Last resort: shrink `--d-model` from 384 → 256.

### `train_ppo.py` KL diverges (> 0.1 sustained)
Either `--kl-init-coef` is too low or `--target-kl` is too loose.
Defaults are 0.05 / 0.02 — bump `--kl-init-coef` to 0.1 if KL keeps
exceeding target.

### `train_rm.py` says "pairs: 0 (train=0, val=0)"
The preferences DB has snippets but no pairwise comparisons. Use
`scripts/serve_feedback.py` to label some, OR seed synthetically
for smoke:

```python
from htbrl.data.preference_dataset import PreferenceStore
store = PreferenceStore("data/preferences.db")
left = store.add_snippet(matrix='enterprise', content_text='...', source='manual')
right = store.add_snippet(matrix='enterprise', content_text='...', source='manual')
store.add_preference(left_id=left, right_id=right, label='left', labeler='you')
store.close()
```

### `eval.py` produces all-zero metrics
Either:
- The checkpoint is too small / undertrained (likely if the smoke run
  produced it).
- The eval suite uses a target the env can't reach.
- The agent's vocab coverage is < 1 (it's outputting the same tool
  every step). Check `vocab_coverage` in the JSON; if it's near 0,
  raise `--entropy-coef` for the next training run.

## Permissions / safety

### "Permission for this action has been denied" from sandboxing
The agent's environment refused a destructive or unauthorised
operation. This is by design. Examples:
- `sudo apt-get install` on the operator's Kali host without explicit
  user authorisation.
- Force-pushing to `main`.
- Sharing files outside the configured allowlist.

If you (the operator) intend to allow the action, run it yourself
or grant the agent explicit permission for that specific call.

### `git push` asks for credentials
The repo doesn't ship credentials and shouldn't store them. Use:

```powershell
git push origin main      # uses cached creds OR opens browser auth
```

If you're trying to push from the agent on your behalf, the agent
must be told explicitly to push (it never pushes spontaneously).

## When the agent is stuck

- Type **what** you tried + **what** the error said. Don't paraphrase.
  Paste the actual stderr / output.
- Tell the agent which **phase** you're on (academy walking? BC
  training? PPO?). Each phase has different failure modes.
- If the agent suggests something that doesn't apply, push back —
  it'll re-check rather than dig in.
