"""HTB Academy module/section -> MITRE ATT&CK technique mapping.

The auto-learner emits Demonstration objects whose turns carry
``techniques_attempted`` / ``techniques_succeeded`` lists of ATT&CK technique
IDs (per project_decisions: ATT&CK is first-class). Without a mapping, every
academy turn would record empty lists, killing ATT&CK coverage metrics for
the academy track.

Design:
- ``ACADEMY_MODULE_TECHNIQUES`` keys on lowercase substrings searched in the
  module's id + title. Multiple keywords may match; the union is returned.
- ``SECTION_TECHNIQUE_HINTS`` narrows a module's technique list to the ones a
  section actually covers, by keyword-matching section.title + body_text. We
  only emit techniques already in the module-level set so a section can never
  invent a technique its module hasn't claimed. Sections with no hit fall
  back to the module-level list (theory pages still inherit coverage).
- All IDs are real ATT&CK Enterprise IDs. No LLMs, no pretrained models -
  pure Python keyword lookup.
"""

from __future__ import annotations

import re

from htbrl.academy.page_models import AcademyModule, AcademySection


# ----- module-level keyword -> [technique IDs] ------------------------------
# Inline ID comments name the technique for human review. Sub-techniques use
# the canonical "Txxxx.yyy" form. We list a small canonical handful per module
# rather than enumerating sub-techniques exhaustively. Empty lists for
# foundational/non-offensive modules are intentional - they keep coverage
# metrics honest.
ACADEMY_MODULE_TECHNIQUES: dict[str, list[str]] = {
    # OS / shell foundations ---
    "linux fundamentals": [
        "T1059.004",  # Unix Shell
        "T1083",      # File and Directory Discovery
        "T1018",      # Remote System Discovery
        "T1057",      # Process Discovery
    ],
    "windows fundamentals": [
        "T1059.001",  # PowerShell
        "T1059.003",  # Windows Command Shell
        "T1083", "T1018", "T1057",
    ],
    # Recon / scanning ---
    "network enumeration": [
        "T1046",      # Network Service Discovery
        "T1018",      # Remote System Discovery
        "T1595.001",  # Active Scanning: Scanning IP Blocks
    ],
    "nmap": ["T1046", "T1018", "T1595.001"],
    "footprinting": [
        "T1595.001",
        "T1595.002",  # Active Scanning: Vuln Scanning
        "T1592.002",  # Gather Victim Host Info: Software
        "T1590.002",  # Gather Victim Network Info: DNS
    ],
    "information gathering": ["T1595.001", "T1592.002", "T1590.002"],
    # Web ---
    "web application": [
        "T1190",      # Exploit Public-Facing Application
        "T1505.003",  # Server Software Component: Web Shell
        "T1136",      # Create Account
    ],
    "web request": ["T1071.001", "T1190"],   # T1071.001 = Web Protocols
    "web attack": ["T1190", "T1505.003"],
    "sql injection": ["T1190", "T1213"],     # T1213 = Data from Info Repos
    "cross-site scripting": ["T1059.007", "T1190"],   # JavaScript
    # Active Directory ---
    "active directory": [
        "T1558",      # Steal or Forge Kerberos Tickets
        "T1003.006",  # OS Credential Dumping: DCSync
        "T1208",      # Kerberoasting (legacy ID; T1558.003 in newer ATT&CK)
    ],
    "kerberos": ["T1558", "T1208"],
    # Binary exploitation ---
    "buffer overflow": [
        "T1203",      # Exploitation for Client Execution
        "T1068",      # Exploitation for Privilege Escalation
    ],
    "binary exploitation": ["T1203", "T1068"],
    "stack-based buffer": ["T1203", "T1068"],
    # Credentials ---
    "password attacks": [
        "T1110",      # Brute Force
        "T1003",      # OS Credential Dumping
    ],
    "hashcat": ["T1110", "T1003"],
    "john the ripper": ["T1110", "T1003"],
    "credential": ["T1003", "T1110"],
    # Shells / RCE ---
    "reverse shell": ["T1059", "T1071.001"],
    "shells": ["T1059", "T1071.001"],
    "shell": ["T1059", "T1071.001"],
    # Privilege escalation ---
    "privilege escalation": [
        "T1068",
        "T1548",      # Abuse Elevation Control Mechanism
    ],
    # File transfer / staging ---
    "file transfer": ["T1105"],   # Ingress Tool Transfer
    # Tunneling / pivoting ---
    "tunneling": ["T1572", "T1090"],   # Protocol Tunneling, Proxy
    "pivot": ["T1572", "T1090"],
    # Malware analysis / obfuscation ---
    "malware analysis": [
        "T1027",      # Obfuscated Files or Information
        "T1140",      # Deobfuscate/Decode Files or Information
    ],
    # Theory-only / non-offensive (keep empty deliberately) ---
    "intro to academy": [],
    "learning process": [],
}


# ----- section narrowing keywords ------------------------------------------
# Each key is lowercased substring searched in section.title + body_text.
# A hint contributes a technique only if it's already in the module-level
# list (intersection), so sections cannot invent unrelated tags.
SECTION_TECHNIQUE_HINTS: dict[str, tuple[str, ...]] = {
    # Linux/Windows fundamentals
    "find files": ("T1083",),
    "find directories": ("T1083",),
    "directory discovery": ("T1083",),
    "file system": ("T1083",),
    "ls ": ("T1083",),
    "process": ("T1057",),
    "ps ": ("T1057",),
    "remote": ("T1018",),
    # Shells
    "powershell": ("T1059.001",),
    "cmd.exe": ("T1059.003",),
    "bash": ("T1059.004",),
    "unix shell": ("T1059.004",),
    "reverse": ("T1059", "T1071.001"),
    # Recon
    "port scan": ("T1046",),
    "service scan": ("T1046",),
    "version detect": ("T1046",),
    "ping sweep": ("T1018", "T1595.001"),
    "host discovery": ("T1018",),
    "ip block": ("T1595.001",),
    # Web
    "web shell": ("T1505.003",),
    "create account": ("T1136",),
    # AD
    "kerberoast": ("T1208",),
    "dcsync": ("T1003.006",),
    "ticket": ("T1558",),
    # Credentials
    "brute force": ("T1110",),
    "dump": ("T1003",),
    "hash": ("T1003", "T1110"),
    # Privesc
    "elevation": ("T1548",),
    "exploit kernel": ("T1068",),
    # Tunneling
    "proxy": ("T1090",),
    "ssh tunnel": ("T1572",),
    "tunnel": ("T1572",),
    # Malware
    "obfuscat": ("T1027",),
    "deobfuscat": ("T1140",),
    "decode": ("T1140",),
    # Buffer overflow
    "shellcode": ("T1203", "T1068"),
    "exploit": ("T1203",),
}


# ----- public API ----------------------------------------------------------

_TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def techniques_for_module(module: AcademyModule) -> list[str]:
    """Return the union of ATT&CK techniques matching the module.

    Both ``id`` and ``title`` are lowercased and substring-matched against
    every key in ``ACADEMY_MODULE_TECHNIQUES``. Returns ``[]`` for modules
    with no offensive keyword match (e.g. "Intro to Academy") - by design,
    not every module teaches an offensive technique.
    """
    haystack = f"{module.id} {module.title}".lower()
    matched: list[str] = []
    for keyword, techniques in ACADEMY_MODULE_TECHNIQUES.items():
        if keyword in haystack:
            matched.extend(techniques)
    return [t for t in _dedup(matched) if _TECHNIQUE_RE.match(t)]


def techniques_for_section(
    section: AcademySection,
    module_techniques: list[str],
) -> list[str]:
    """Narrow a module's technique list to those a section actually covers.

    Searches section.title + body_text for ``SECTION_TECHNIQUE_HINTS`` keys.
    A hint only contributes IDs already in ``module_techniques`` so we never
    invent a technique the module hasn't claimed. With no hint matches we
    fall back to the full module list - theory-only sections still get
    credited rather than dropping to ``[]``.
    """
    if not module_techniques:
        return []
    module_set = set(module_techniques)
    haystack = f"{section.title}\n{section.body_text}".lower()

    biased: list[str] = []
    for keyword, candidates in SECTION_TECHNIQUE_HINTS.items():
        if keyword in haystack:
            biased.extend(c for c in candidates if c in module_set)

    return _dedup(biased) if biased else list(module_techniques)
