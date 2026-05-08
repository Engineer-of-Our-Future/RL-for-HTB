# Training pipeline

Every training + eval script in this repo runs end-to-end. This doc is
the operator's cookbook: run the commands top-to-bottom and you go
from a fresh checkout to an evaluable agent. The commands here are
the same ones in `LABS_PLAN.md` but separated into reproducible
recipes.

Sister docs:
- `docs/architecture.md` — what each script owns and how data flows.
- `docs/labs_walking.md` — collecting labs demos, the manual half of Phase 5.
- `docs/academy_walking.md` — the auto-learner half of Phase 5b.
- `docs/troubleshooting.md` — when things break.

## 0. One-time bootstrap

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts\smoke_test.py        # GPU + (optional) Kali SSH check
pytest -q                            # 530 collected, all green
```

If the SSH check skipped, that's expected — set `HTBRL_KALI_HOST`
before running training that needs the env wrapper.

## 1. Train the BPE tokenizer (Phase 2)

One-time. The tokenizer's vocab is frozen once written.

```powershell
python scripts\train_tokenizer.py `
  --output tokenizer\v1.json `
  --vocab-size 32768 `
  --corpus-dir data\corpus `
  --min-frequency 2
```

If `data/corpus/` is empty, the script seeds it from the tool registry
templates. For a real run, populate it with shell histories, nmap
output samples, HTB writeup text first.

**Sanity:** `pytest tests/tokenizer/ -q` (27 tests).

## 2. Collect demos

Two parallel sources feed BC:

### 2a. Academy auto-learner (Phase 5b)

```powershell
# Launch Chrome with CDP debugging
.\scripts\start_chrome_for_htb.ps1

# In Chrome, log into HTB Academy manually (Cloudflare-friendly).
# Then run the auto-learner against an owned module:
python scripts\htb_academy_run.py --module-id 18

# Or interactively via the wizard (operator approves each answer):
python scripts\htb_academy_wizard.py --module-id 18
```

Each run writes one demo to `data/auto_demos/academy_module_<id>.msgpack.gz`.

### 2b. Manual lab walk (Phase 5)

```powershell
$env:HTBRL_KALI_HOST     = 'claude@192.168.1.219:22'
$env:HTBRL_KALI_PASSWORD = 'claude'

python scripts\collect_demos.py `
  --target-id htb-starting-point:meow `
  --kali-host $env:HTBRL_KALI_HOST
```

The wizard prompts you for each tool and slot per turn; demos land in
`data/demos/wizard_<target>_<ts>.msgpack.gz`. Aim for 10-30 lab demos
before serious BC training.

**Coverage check:**

```powershell
python scripts\academy_coverage.py
python scripts\coverage_report.py     # mixed academy + lab demos
```

## 3. BC pretraining (Phase 5)

Smoke run (CPU, tiny model, sanity check):

```powershell
python scripts\train_bc.py `
  --demo-root data\auto_demos `
  --tokenizer-path tokenizer\v1.json `
  --epochs 1 --max-steps 5 --batch-size 2 `
  --d-model 64 --n-layers 2 --n-heads 2 --d-ff 128 `
  --max-seq-len 64 --device cpu `
  --output .local\bc_smoke.pt
```

Mid-size smoke (GPU, ~10 min on 3060):

```powershell
python scripts\train_bc.py `
  --demo-root data\auto_demos `
  --demo-root data\demos `
  --tokenizer-path tokenizer\v1.json `
  --epochs 3 --batch-size 8 --max-steps 60 `
  --d-model 128 --n-layers 4 --n-heads 4 --d-ff 512 `
  --max-seq-len 256 --device cuda `
  --output checkpoints\bc-mixed-smoke.pt
```

Real run (GPU, full size — multi-root mixes academy + lab demos):

```powershell
python scripts\train_bc.py `
  --demo-root data\auto_demos `
  --demo-root data\demos `
  --tokenizer-path tokenizer\v1.json `
  --epochs 50 --batch-size 32 `
  --d-model 384 --n-layers 8 --n-heads 8 --d-ff 1536 `
  --max-seq-len 1024 --device cuda `
  --label-smoothing 0.05 `
  --lr 1e-4 `
  --num-workers 4 `
  --output checkpoints\bc-mixed-v1.pt
```

Wall-clock: ~24h on the 3060 per `PLAN.md` budget. Pass criterion:
**held-out tool_acc ≥ 0.6**.

## 4. PPO smoke (Phase 6)

Verify the trainer round-trips before spending GPU days:

```powershell
python scripts\train_ppo.py `
  --env-type stub `
  --n-envs 2 --n-steps 16 `
  --total-rollouts 2 `
  --d-model 64 --n-layers 2 --n-heads 2 --d-ff 128 `
  --max-seq-len 64 --vocab-size 1024 `
  --minibatch-size 16 --n-epochs 1 `
  --kl-init-coef 0 `
  --device cpu `
  --output .local\ppo_smoke
```

Watch for `loss=…` and `ev=…` (explained variance) trending up.

## 5. Single-target PPO against HTB (Phase 6 + Phase 4)

Once `bc-mixed-v1.pt` is ready and a starting-point box is spawned on
HTB:

```powershell
$env:HTBRL_KALI_HOST     = 'claude@192.168.1.219:22'
$env:HTBRL_KALI_PASSWORD = 'claude'

python scripts\train_ppo.py `
  --env-type htb `
  --kali-host $env:HTBRL_KALI_HOST `
  --allowlist-cidr 10.10.10.0/24 `
  --ref-checkpoint checkpoints\bc-mixed-v1.pt `
  --total-rollouts 100 `
  --n-envs 4 --n-steps 128 `
  --kl-init-coef 0.05 --target-kl 0.02 `
  --entropy-coef 0.02 `
  --output runs\ppo-meow-v1
```

**Pass criterion:** foothold > 0% within 50 rollouts. If not, demo
set isn't rich enough — collect more (step 2) and retry.

## 6. Reward model (Phase 8)

### 6a. Label preferences

```powershell
python scripts\serve_feedback.py --db data\preferences.db --port 8765
# Open http://localhost:8765, click "left/right/tie" on snippet pairs.
# 100-300 pairs is the minimum useful set; 1500+ for a good RM.
```

### 6b. Train the RM

```powershell
python scripts\train_rm.py `
  --db data\preferences.db `
  --tokenizer-path tokenizer\v1.json `
  --d-model 256 --n-layers 4 --n-heads 4 --d-ff 1024 `
  --max-seq-len 512 `
  --epochs 20 --batch-size 16 `
  --val-fraction 0.1 `
  --device cuda `
  --output checkpoints\rm-v1.pt
```

**Pass criterion:** held-out pairwise accuracy ≥ 0.65.

## 7. RLHF run (Phase 9)

Combines env reward + RM + RND + KL into the composite signal.

```powershell
python scripts\train_ppo.py `
  --env-type htb `
  --kali-host $env:HTBRL_KALI_HOST `
  --allowlist-cidr 10.10.10.0/24 `
  --ref-checkpoint checkpoints\bc-mixed-v1.pt `
  --rm-checkpoint checkpoints\rm-v1.pt `
  --total-rollouts 50000 --n-envs 8 --n-steps 128 `
  --kl-init-coef 0.05 --target-kl 0.05 `
  --entropy-coef 0.01 `
  --use-8bit-adamw `
  --compile --compile-mode reduce-overhead `
  --output runs\rlhf-v1
```

Wall-clock: 3-7 days for 5-10M env steps. The flags
`--use-8bit-adamw / --compile` come from Phase 10's hardware-targeted
optimisations.

## 8. Eval (Phase 11)

```powershell
python scripts\eval.py `
  --checkpoint runs\rlhf-v1\ckpt-final.pt `
  --tokenizer-path tokenizer\v1.json `
  --suite configs\eval\holdout_v1.yaml `
  --n-envs 4 --n-episodes-per-env 5 `
  --max-steps 200 `
  --device cuda `
  --output-json runs\rlhf-v1\eval.json
```

Reports foothold rate, user-flag rate, root-flag rate, MITRE
technique coverage, average kill-chain depth.

**Final pass criterion (per `PLAN.md` Phase 11):**
- Foothold rate ≥ 60% on the held-out easy boxes
- User-flag rate ≥ 30%
- Vocab coverage > 50% (no policy collapse)

## 9. Inspect a trained run

```powershell
# View the run's training log:
type runs\rlhf-v1\log.jsonl | head

# List checkpoints with metadata:
python scripts\list_checkpoints.py runs\rlhf-v1\

# Render a specific demo or rollout:
python -c "from htbrl.data.demo_dataset import load_demonstration; d=load_demonstration(r'data\demos\wizard_meow_*.msgpack.gz'); print(d.outcome)"
```

## When to skip steps

- **Skip step 6 (RM)** if you're staying with auto-rewards only. Set
  `--rm-checkpoint ""` in step 7 and the trainer falls back to env
  reward + RND + KL (i.e. PPO without the H in RLHF). Faster but
  the agent can't learn from "this command was risky / noisy /
  inelegant" preferences.
- **Skip step 8 (eval)** at peril — without it you can't ship.

## Common pitfalls

See `docs/troubleshooting.md`.
