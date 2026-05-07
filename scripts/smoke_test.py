r"""Phase 0 smoke test.

Verifies that the dev environment is wired up end-to-end:
  1. PyTorch imports and (ideally) sees the 3060.
  2. SSH to the Kali attacker works (optional - skipped if HTBRL_KALI_HOST unset).

Exit code is 0 only when all non-skipped checks pass. Run from the project root:

    python scripts/smoke_test.py

To enable the SSH check:

    HTBRL_KALI_HOST="kali@127.0.0.1:2222" python scripts/smoke_test.py     # bash
    $env:HTBRL_KALI_HOST = "kali@127.0.0.1:2222"; python scripts\smoke_test.py    # PowerShell

Auth: prefers a private key via HTBRL_KALI_KEY (path to identity file).
Falls back to HTBRL_KALI_PASSWORD if no key is set. Password auth is a
convenience for dev setups; key auth is the recommended path for any
shared / persistent Kali host.
"""


from __future__ import annotations

import os
import sys


def _check_torch() -> bool:
    try:
        import torch
    except ImportError as exc:
        print(f"[FAIL] torch import: {exc}")
        return False

    print(f"[ OK ] torch {torch.__version__}")

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available - training will fall back to CPU.")
        print("       On Windows + RTX 3060, install a CUDA-enabled torch wheel:")
        print("         pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return True

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cap = torch.cuda.get_device_capability(0)
    print(f"[ OK ] GPU:  {name}")
    print(f"[ OK ] VRAM: {vram_gb:.1f} GB")
    print(f"[ OK ] Compute capability: {cap[0]}.{cap[1]}")

    if vram_gb < 11.0:
        print(f"[WARN] Less than 11 GB VRAM detected ({vram_gb:.1f}). PLAN.md was budgeted for a 12 GB 3060.")

    # Tiny tensor round-trip to confirm the runtime is actually usable.
    x = torch.randn(64, 64, device="cuda")
    y = (x @ x.t()).sum().item()
    if y != y:  # NaN guard
        print("[FAIL] CUDA matmul produced NaN.")
        return False
    print("[ OK ] CUDA matmul round-trip")
    return True


def _check_ssh() -> bool:
    host_env = os.environ.get("HTBRL_KALI_HOST")
    if not host_env:
        print("[SKIP] HTBRL_KALI_HOST not set - skipping SSH check.")
        print("       Example: HTBRL_KALI_HOST=htbrl@127.0.0.1:2222")
        return True

    user_at_host, _, port_str = host_env.partition(":")
    user, _, hostname = user_at_host.partition("@")
    if not (user and hostname):
        print(f"[FAIL] HTBRL_KALI_HOST must be 'user@host[:port]'; got {host_env!r}")
        return False
    try:
        port = int(port_str) if port_str else 22
    except ValueError:
        print(f"[FAIL] non-numeric port in HTBRL_KALI_HOST: {port_str!r}")
        return False

    try:
        import paramiko
    except ImportError as exc:
        print(f"[FAIL] paramiko import: {exc}")
        return False

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        "hostname": hostname,
        "port": port,
        "username": user,
        "timeout": 5,
        "allow_agent": True,
        "look_for_keys": True,
    }
    key_path = os.environ.get("HTBRL_KALI_KEY")
    password = os.environ.get("HTBRL_KALI_PASSWORD")
    if key_path:
        # Explicit private key (e.g. ~/.ssh/htbrl_kali for the WSL Kali setup).
        connect_kwargs["key_filename"] = os.path.expanduser(key_path)
    elif password:
        # Fallback: password auth from env (convenient for dev setups
        # where the operator's Kali laptop uses password login). Disable
        # agent + key lookup so paramiko doesn't try keys first and
        # surface a confusing AuthenticationException when no key matches.
        connect_kwargs["password"] = password
        connect_kwargs["allow_agent"] = False
        connect_kwargs["look_for_keys"] = False

    try:
        client.connect(**connect_kwargs)
    except Exception as exc:  # paramiko raises a wide tree
        print(f"[FAIL] SSH connect to {host_env}: {exc}")
        return False

    try:
        _, stdout, stderr = client.exec_command("whoami && uname -a", timeout=5)
        whoami = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
    finally:
        client.close()

    if not whoami:
        print(f"[FAIL] SSH ran but produced no whoami output. stderr={err!r}")
        return False
    print(f"[ OK ] SSH whoami -> {whoami}")
    return True


def main() -> int:
    print("--- Phase 0 smoke test ---")
    checks = [_check_torch(), _check_ssh()]
    print("--- DONE ---" if all(checks) else "--- FAILED ---")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
