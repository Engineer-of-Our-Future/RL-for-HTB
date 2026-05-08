# LABS_PLAN — Real-target training

This is the second plan after `PLAN.md`'s Phase 5b academy auto-learner.
The academy half teaches the agent the *language* of pentesting (read
modules, learn cheat-sheet rows, attempt sandbox questions). The labs
half teaches it to *operate* — picking, sequencing, and reasoning over
tool calls against real attacker→target episodes.

## Where we are when we start the labs plan

- ✅ `PLAN.md` Phases 0–11 code complete; every script smoke-runs.
- ✅ `HTBEnv` works against a Kali attacker over SSH.
- ✅ Academy probes (HTTP / LFI / SSH / RFI) wired and live-verified.
- 🟡 5 academy modules walked → 15 demos in `data/auto_demos/`. Coverage grows operator-paced.
- 🟡 No real-scale RL training yet (BC ran 5 steps for smoke; PPO ran 2 rollouts).

## What "labs" needs that academy doesn't

| Concern | Academy solution | Labs need |
|---|---|---|
| Target IP source | Per-section spawn from CDP panel | Operator spawns lab box on HTB; we get IP from VPN |
| Network reach | External targets reachable from any host | Lab targets only reachable through HTB OpenVPN |
| Episode length | Single section answer (~1-3 turns) | Full kill-chain, can be 30-200+ turns |
| Reward signal | Question accept/reject + cube delta | Foothold / user-flag / root-flag captures + step penalty |
| Reset semantics | Re-spawn academy target | HTB doesn't allow resets; rotate boxes instead |
| Allowlist | Public academy IPs (154.57.x.x, 10.129.x.x) | HTB starting-point CIDR (`10.10.10.0/24`, etc.) |

## The 6 labs-plan steps

### 1. Operator-side VPN setup

Connect Kali to HTB labs OpenVPN (separate from academy). One-time:

```bash
# on the Kali attacker (laptop or WSL):
sudo openvpn --config ~/lab_X.ovpn --daemon --log /tmp/htblab-openvpn.log
# verify:
ip -4 addr show tun0     # expect 10.10.x.x
ping -c 2 10.10.10.1     # HTB gateway
```

Update `configs/env/htb.yaml` with the right `allowlist_cidrs` for the lab range.

### 2. Manual demo collection (small, high-quality set)

Use `scripts/collect_demos.py` to walk 10-20 starting-point boxes manually.
Each demo captures (state, tool_id, slots, render, reward) tuples.
Quality > quantity: pick boxes you understand, narrate every turn.

```powershell
python scripts/collect_demos.py `
  --target-id htb-starting-point:meow `
  --kali-host claude@192.168.1.219:22 `
  --kali-key $HOME\.ssh\htbrl_kali
```

Outputs land in `data/demos/wizard_*.msgpack.gz` (mixed in with the
academy demos by `train_bc.py`).

### 3. BC pre-training at full size

Once demo set is ≥30 trajectories (15 academy + 15 lab), run BC at the
PLAN.md target size (`d_model=384`, 8 layers, ~50–80M params). This is
the agent's first real model.

```powershell
python scripts/train_bc.py `
  --demo-root data/auto_demos `        # academy demos
  --demo-root data/demos `              # lab demos (mix in)
  --tokenizer-path tokenizer/v1.json `
  --epochs 50 --batch-size 32 `
  --d-model 384 --n-layers 8 --n-heads 8 --d-ff 1536 `
  --max-seq-len 1024 --device cuda `
  --output checkpoints/bc-mixed-v1.pt
```

Wall-clock: ~24 hours on the 3060 (per PLAN.md budget). Verify
**tool_acc ≥ 0.6** on a held-out demo before moving on.

### 4. Single-target PPO smoke

Pick ONE starting-point box, spawn it on HTB, verify the env reaches
it via VPN, then run PPO for a few hours. The point is to confirm:
- Allowlist gate fires on outside-CIDR IPs.
- nmap / gobuster / hydra parsers populate `obs_text` with structured features.
- Reward shaping (per `htbrl.env.rewards`) credits foothold / flag.
- KL stays near `--target-kl` against the BC reference.

```powershell
$env:HTBRL_KALI_HOST     = 'claude@192.168.1.219:22'
$env:HTBRL_KALI_PASSWORD = 'claude'

python scripts/train_ppo.py `
  --env-type htb `
  --kali-host $env:HTBRL_KALI_HOST `
  --allowlist-cidr 10.10.10.0/24 `
  --ref-checkpoint checkpoints/bc-mixed-v1.pt `
  --total-rollouts 50 --n-envs 4 --n-steps 128 `
  --kl-init-coef 0.05 --target-kl 0.02 `
  --output runs/ppo-meow-v1
```

**Pass criterion:** the agent gets foothold > 0% on the chosen box
within ~50 rollouts. If not, demo set isn't rich enough — return to
step 2 with more demos.

### 5. Reward-model bootstrap (Phase 8 → 9)

In parallel with single-target PPO, the operator labels preference
pairs via the FastAPI UI:

```powershell
python scripts/serve_feedback.py --db data/preferences.db
# open http://localhost:8765, label 100-300 pairs
```

When you have ≥300 pairs, train the RM:

```powershell
python scripts/train_rm.py `
  --db data/preferences.db `
  --tokenizer-path tokenizer/v1.json `
  --d-model 256 --n-layers 4 `
  --epochs 20 --batch-size 16 --device cuda `
  --output checkpoints/rm-v1.pt
```

**Pass criterion:** held-out pairwise accuracy ≥ 0.65 (random = 0.5).

### 6. Multi-target RLHF run

Final phase — PPO with composite reward (env reward + RM + RND + KL),
rotating across a pool of starting-point boxes. This is the
"production" run.

```powershell
python scripts/train_ppo.py `
  --env-type htb `
  --target-pool configs/env/htb_starting_pool.yaml `
  --rm-checkpoint checkpoints/rm-v1.pt `
  --ref-checkpoint checkpoints/bc-mixed-v1.pt `
  --total-rollouts 50000 --n-envs 8 --n-steps 128 `
  --kl-init-coef 0.05 --target-kl 0.05 `
  --output runs/rlhf-v1
```

Wall-clock: 3–7 days for 5–10M env steps. Eval gates the run:

```powershell
python scripts/eval.py `
  --checkpoint runs/rlhf-v1/ckpt-final.pt `
  --suite configs/eval/holdout_v1.yaml `
  --output-json runs/rlhf-v1/eval.json
```

**Final pass criterion (per `PLAN.md` Phase 11):**
- Foothold rate ≥ 60% on the held-out easy boxes
- User-flag rate ≥ 30%
- Vocab coverage > 50% (no policy collapse)
- KL-from-BC < 0.1 (no catastrophic drift)

## What unlocks each step

| Step | Blocker |
|---|---|
| 1. VPN setup | Operator runs OpenVPN against an HTB Pwnbox `.ovpn` file |
| 2. Demo collection | Step 1 done; operator's manual time |
| 3. BC pretraining | Step 2 yields ≥30 demos; ~24h GPU time |
| 4. Single-target PPO | Step 3 produces a checkpoint; HTB box spawned |
| 5. RM training | Operator labels ≥300 preferences via the UI |
| 6. RLHF run | Steps 3 + 5 done; 3-7 days GPU time |

## Out of scope

- **Solving the 3 stuck academy questions** (mod 33 sec 518 chattr SQLi,
  mod 23 sec 1494 view= LFI, mod 23 sec 513 sumace) — academy auto-learner
  may resolve them later when probe library grows; not a labs-plan blocker.
- **HTB Pro Labs / Endgames** — too large for the 3060's training budget.
  Starting Point + Easy is the target.
- **Mobile / ICS attacker tracks** — `PLAN.md` Phases 13-17. Out of scope
  for this labs-plan; revisited after the Enterprise track is shippable.

## When labs-plan is done

The project is **ready** when:
1. All 6 steps above completed end-to-end.
2. Eval JSON shows the Phase 11 thresholds (foothold ≥ 60%, user-flag ≥ 30%).
3. The end-to-end command pipeline (BC → preference labelling → RM → RLHF → eval)
   has been demonstrated as a single reproducible run, documented in `docs/training_pipeline.md`.

Then `README.md`'s status line gets bumped to **"Phase 12 — packaged & runnable"**.
