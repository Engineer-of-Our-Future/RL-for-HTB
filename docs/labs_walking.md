# Labs walking — manual demo collection

Once the academy auto-learner has produced a baseline of demos
(`data/auto_demos/*.msgpack.gz`), the next demos come from manual
walks against real HTB lab boxes. Each walk produces one
`Demonstration` in `data/demos/` that BC training mixes in
alongside the academy half.

This is `LABS_PLAN.md` step 2, expanded.

## When to start

- ✅ Academy probes wired (HTTP / LFI / SSH / RFI) — `0fd6381` + `52963e3`.
- ✅ HTBEnv works end-to-end against the operator's Kali.
- ✅ At least one academy module finished so the cube gate is unblocked.

## VPN setup (one-time)

The lab boxes live behind HTB's VPN. Connect Kali first (laptop or WSL):

```bash
# on the Kali attacker
sudo openvpn --config ~/lab.ovpn --daemon --log /tmp/htblab-openvpn.log
ip -4 addr show tun0     # expect 10.10.x.x
ping -c 2 10.10.10.1     # HTB gateway
```

Update `configs/env/htb.yaml`:

```yaml
allowlist_cidrs:
  - "10.10.10.0/24"      # starting point
  - "10.10.11.0/24"      # easy machines pool
```

The env wrapper refuses any rendered command whose extracted IPs fall
outside these CIDRs. This is intentional — it's the safety guardrail
that stops an exploring policy from accidentally scanning the public
internet during rollouts.

## Spawn a starting-point box

Pick one from htb's "Starting Point" tier (Meow, Fawn, Dancing, …).
Spawn it via the HTB UI; note the box IP (something like 10.10.11.42).

## Walk one episode

```powershell
$env:HTBRL_KALI_HOST     = 'claude@192.168.1.219:22'
$env:HTBRL_KALI_PASSWORD = 'claude'

python scripts\collect_demos.py `
  --target-id htb-starting-point:meow `
  --kali-host $env:HTBRL_KALI_HOST `
  --output-dir data\demos
```

The wizard interactively walks you through:

1. Type a bash command (e.g. `nmap -sS -p- 10.10.11.42`) at the `>` prompt.
2. The command runs over SSH; the wizard shows the (truncated) output.
3. The wizard suggests the top-3 tools whose `command_template` looks
   like what you typed; you pick one (or `manual` to log free-text,
   or `skip` to drop the turn).
4. The wizard prompts for each slot value.
5. Reward — type a number or an alias:
   - `step` → -0.01
   - `timeout` → -0.1
   - `blocked` → -0.5 (allowlist violation)
   - `new_port` → +0.1
   - `new_service` → +0.2
   - `user_shell` → +0.5
   - `user_flag` → +1.0
   - `root_shell` → +1.5
   - `root_flag` → +2.0
6. Repeat. Type `done` when the episode ends, or `abort` to discard.
7. Wizard asks for outcome flags (foothold / user_flag / root_flag),
   writes a `.msgpack.gz` to `data/demos/`.

## Demo quality > demo quantity

Each walk should be a *clean* episode you understand turn-by-turn:

- Don't log noisy commands you ran for personal exploration. Use
  `skip` on those.
- Match each tool to its closest registry entry. If nothing matches,
  use `manual` (the demo carries the raw command but no
  `tool_id`/`techniques_attempted`).
- Tag every turn's reward correctly. The reward shaping signal is
  what BC + PPO learn from.
- Capture the full kill chain: recon → enumeration → exploitation →
  privilege escalation → post-exploitation. A 30-turn demo with a
  user flag at the end is far more valuable than 100 random nmap
  invocations.

## Target list (suggested)

Start with these Starting Point boxes (well-documented, simple kill chains):

| Box | Difficulty | Skills |
|---|---|---|
| Meow | Trivial | telnet enumeration, default creds |
| Fawn | Trivial | anonymous FTP |
| Dancing | Trivial | SMB anonymous |
| Redeemer | Trivial | Redis enumeration |
| Appointment | Easy | SQLi (auth bypass) |
| Sequel | Easy | MySQL anonymous + flag in DB |
| Crocodile | Easy | FTP cred + admin login |
| Responder | Easy | LFI + Responder NTLM relay |
| Three | Easy | S3 bucket + webshell upload |
| Funnel | Easy | FTP enumeration + Postgres tunnel |

Walk all 10 → ~10-15 demos. Combined with the academy demos you should
have enough for a first BC pass.

## After demos: BC training

See `docs/training_pipeline.md` step 3. The academy demos
(`data/auto_demos/`) and lab demos (`data/demos/`) are passed
together via repeated `--demo-root` flags:

```powershell
python scripts\train_bc.py `
  --demo-root data\auto_demos `
  --demo-root data\demos `
  --tokenizer-path tokenizer\v1.json `
  --epochs 50 `
  --d-model 384 --n-layers 8 ... `
  --output checkpoints\bc-mixed-v1.pt
```

## Things that go wrong

- **Allowlist violation** — the env refused your command because an IP
  outside the configured CIDRs appeared in the rendered command.
  Update `configs/env/htb.yaml` or fix the slot.
- **SSH session drops** — Kali went idle / VPN dropped. Re-establish
  with `wsl --shutdown && wsl -d kali-linux` (WSL) or restart the
  laptop's openvpn daemon.
- **Tool suggestions all wrong** — your command's first token doesn't
  map to a registered tool. Either use `manual` or add the tool to
  the registry under `src/htbrl/tools/registry/`.

See `docs/troubleshooting.md` for the longer list.
