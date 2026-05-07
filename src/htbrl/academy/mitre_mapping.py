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
    "obfuscat": ["T1027", "T1140"],         # "obfuscation" / "obfuscated"
    "deobfuscat": ["T1027", "T1140"],
    "javascript deobfusc": ["T1027", "T1140", "T1059.007"],
    # Theory-only / non-offensive (keep empty deliberately) ---
    "intro to academy": [],
    "learning process": [],
    "setting up": [],
    "documentation & reporting": [],
    "documentation and reporting": [],
    "incident handling process": [],
    "bug bounty hunting process": [],
    "penetration testing process": [],
    # ---- web vulns ----------------------------------------------------------
    "file inclusion": ["T1190", "T1505.003"],
    "file upload attacks": ["T1190", "T1505.003"],
    "command injection": ["T1059", "T1190"],
    "command injections": ["T1059", "T1190"],
    "server-side attacks": ["T1190", "T1505.003"],
    "session security": ["T1539", "T1550.004"],   # Steal Web Session Cookie / Web Session Cookie
    "broken authentication": ["T1078", "T1110"],   # Valid Accounts / Brute Force
    "attacking authentication mechanisms": ["T1110", "T1078", "T1556"],
    "deserialization": ["T1190", "T1059"],
    "injection attacks": ["T1190", "T1059"],
    "whitebox attacks": ["T1190", "T1059"],
    "whitebox pentesting": ["T1190", "T1059"],
    "http attacks": ["T1190"],
    "https/tls attacks": ["T1557", "T1040"],   # Adversary-in-the-Middle / Network Sniffing
    "tls attack": ["T1557", "T1040"],
    "abusing http": ["T1190"],
    "http misconfigurations": ["T1190"],
    "web service & api attacks": ["T1190", "T1071.001"],
    "api attack": ["T1190", "T1071.001"],
    "web fuzzing": ["T1190", "T1595.003"],     # Active Scanning: Wordlist Scanning
    "ffuf": ["T1190", "T1595.003"],
    "attacking web applications with ffuf": ["T1190", "T1595.003"],
    "introduction to web applications": ["T1190", "T1505.003"],
    "hacking wordpress": ["T1190", "T1505.003"],
    "wordpress": ["T1190"],
    "sqlmap": ["T1190", "T1213"],
    "using web proxies": ["T1090", "T1557"],
    "web proxies": ["T1090", "T1557"],
    "secure coding 101": ["T1190"],
    # ---- enumeration / recon -----------------------------------------------
    "dns enumeration": ["T1018", "T1590.002"],
    "osint": ["T1593", "T1591", "T1589"],     # Search Open Websites/Domains, Gather Victim Org Info, Identity Info
    "corporate recon": ["T1593", "T1591", "T1589"],
    "vulnerability assessment": ["T1595.002"],   # Active Scanning: Vuln Scanning
    "network foundations": ["T1018", "T1046"],
    "introduction to networking": ["T1018", "T1046", "T1590"],
    # ---- credential access / brute force -----------------------------------
    "login brute forcing": ["T1110", "T1110.001", "T1110.003"],   # Password Guessing / Spraying
    "brute force": ["T1110"],
    # ---- exploitation frameworks -------------------------------------------
    "metasploit": ["T1059", "T1190", "T1068"],
    "using the metasploit": ["T1059", "T1190"],
    # ---- traffic / network analysis ----------------------------------------
    "intro to network traffic analysis": ["T1040"],   # Network Sniffing
    "network traffic analysis": ["T1040"],
    "wireshark": ["T1040"],
    "tcpdump": ["T1040"],
    # ---- wireless ----------------------------------------------------------
    "wired equivalent privacy": ["T1040", "T1110"],
    "wep attack": ["T1040", "T1110"],
    "wi-fi protected setup": ["T1110"],
    "wps": ["T1110"],
    "wifi": ["T1040"],
    # ---- platforms ---------------------------------------------------------
    "macos fundamentals": [
        "T1059.004",   # Unix Shell (zsh/bash)
        "T1083", "T1018", "T1057",
    ],
    "android fundamentals": [
        # Mobile ATT&CK ids (mobile matrix overlap with the enterprise track).
        "T1404",       # Exploit OS Vulnerability (Mobile)
        "T1418",       # Software Discovery (Mobile)
        "T1471",       # Data Encrypted for Impact (Mobile)
    ],
    # ---- assembly / binary -------------------------------------------------
    "intro to assembly language": ["T1203", "T1068"],
    "assembly language": ["T1203", "T1068"],
    "stack-based buffer overflows on linux": ["T1203", "T1068"],
    "stack-based buffer overflows on windows": ["T1203", "T1068"],
    # ---- AD-adjacent + post-ex tooling -------------------------------------
    "using crackmapexec": ["T1110", "T1003", "T1021.002"],   # SMB/Windows Admin Shares
    "crackmapexec": ["T1110", "T1003", "T1021.002"],
    "windows attacks & defense": ["T1558", "T1003", "T1078"],
    "attacking enterprise networks": ["T1558", "T1003", "T1003.006", "T1208"],
    # ---- attacking common targets ------------------------------------------
    "attacking common applications": ["T1190", "T1505.003"],
    "attacking common services": ["T1110", "T1021"],   # Remote Services
    # ---- introduction to information security ------------------------------
    "introduction to information security": [],   # foundational, non-offensive
    "intro to information security": [],
    # ---- specialty ---------------------------------------------------------
    "introduction to bash scripting": ["T1059.004"],
    "introduction to python": ["T1059.006"],   # Python
    "python 3": ["T1059.006"],
    "introduction to windows command line": ["T1059.003"],
    "windows command line": ["T1059.003"],
    "intro to academy's purple modules": [],
    "purple modules": [],
    "fundamentals of ai": [],
    "applications of ai in infosec": [],
    "brief intro to hardware attacks": ["T1200"],   # Hardware Additions
    "hardware attacks": ["T1200"],
    "game hacking": ["T1055"],   # Process Injection (game memory editing)
    "game reversing": ["T1027", "T1140"],
    "getting started": ["T1595.001", "T1018", "T1046"],   # foundational pentest module
    # ---- defensive / SOC / blue-team ---------------------------------------
    # ATT&CK technique IDs used here are the offensive techniques the
    # defender is *detecting*. The demo turns are observation-side
    # (analyzing artifacts, alerts, logs) so we tag the techniques the
    # learner is being trained to recognize.
    "security monitoring": ["T1059", "T1078", "T1003"],
    "siem fundamentals": ["T1059", "T1078", "T1003"],
    "splunk": ["T1059", "T1003", "T1078"],
    "elastic": ["T1059", "T1003"],
    "threat hunting": ["T1059", "T1003", "T1078"],
    "windows event logs": ["T1059.001", "T1059.003", "T1078"],
    "log sources": ["T1059", "T1078"],
    "yara": ["T1027", "T1140"],
    "sigma for soc": ["T1059", "T1078"],
    "yara & sigma": ["T1027", "T1140", "T1059"],
    "ids/ips": ["T1071.001", "T1027"],
    "intrusion detection": ["T1071.001"],
    "introduction to digital forensics": ["T1003", "T1027"],
    "linux forensics": ["T1059.004", "T1003"],
    "user behavior forensics": ["T1078"],
    "security incident reporting": [],   # process, not technique
    "soc analyst": ["T1059", "T1003"],
    "detecting windows attacks": ["T1059.001", "T1059.003", "T1003"],
    # ---- advanced offensive / AD post-ex ------------------------------------
    "dacl attacks": ["T1078", "T1098"],   # Valid Accounts / Account Manipulation
    "ntlm relay": ["T1557.001", "T1003.001"],   # AitM: LLMNR/NBT-NS / LSASS Memory
    "adcs attacks": ["T1649", "T1558"],   # Steal/Forge Auth Cert / Kerberos Ticket
    "c2 operations": ["T1071.001", "T1071.004", "T1095"],   # Web/DNS/Non-App-Layer Protocol
    "intro to c2": ["T1071.001"],
    "sliver": ["T1071.001"],
    # ---- supply chain / specialty ------------------------------------------
    "supply chain attack": ["T1195"],   # Supply Chain Compromise
    "modern web exploitation": ["T1190", "T1505.003"],
    "advanced xss": ["T1190", "T1059.007"],
    "csrf": ["T1190"],
    "xss": ["T1190", "T1059.007"],
    "parameter logic bugs": ["T1190"],
    # ---- mobile (Mobile ATT&CK matrix) -------------------------------------
    "android application static": ["T1418", "T1623"],   # Software Discovery / Command & Scripting Interpreter (Mobile)
    "android application dynamic": ["T1418", "T1623"],
    "wi-fi penetration testing": ["T1040", "T1110"],
    # ---- programming-language modules (foundational for scripting) ---------
    "introduction to c#": [],   # language module, no specific technique
    # ---- additional general / non-offensive --------------------------------
    "introduction to information security": [],
    "intro to information security": [],
    # ---- advanced offensive (round 3) --------------------------------------
    "windows evasion": ["T1027", "T1055", "T1562"],   # Obfuscate / Process Injection / Impair Defenses
    "evasion technique": ["T1027", "T1055", "T1562"],
    "access token manipulation": ["T1134"],   # Access Token Manipulation
    "binary fuzzing": ["T1203", "T1059"],
    "windows lateral movement": ["T1021", "T1021.002", "T1550"],   # Remote Services / SMB / Use Alternate Auth Material
    "lateral movement": ["T1021", "T1550"],
    "malicious document analysis": ["T1204.002", "T1027"],   # User Execution: Malicious File / Obfuscation
    "mssql": ["T1059.005", "T1190"],   # Visual Basic + SQL exec
    "exchange": ["T1190", "T1078"],
    "sccm attack": ["T1078", "T1190"],
    "attacking graphql": ["T1190", "T1213"],
    "graphql": ["T1190"],
    "android penetration testing": ["T1418", "T1623"],
    "attacking wpa": ["T1110", "T1040"],
    "wpa/wpa2": ["T1110", "T1040"],
    "evil twin": ["T1557", "T1110"],   # AitM
    "captive portal": ["T1557", "T1110"],
    "android forensics": ["T1003"],
    "ai data attack": ["T1565"],   # Data Manipulation
    "red teaming ai": [],   # process/methodology, not specific technique
    "introduction to penetration testing": ["T1595", "T1018", "T1046"],
    "winddbg": ["T1622"],   # Debugger Evasion (kept loose; it's a tool)
    "windbg": ["T1622"],
    "detection & opsec": ["T1027", "T1562"],
    "opsec cyber range": ["T1027", "T1562"],
    # Catch-all defensive flag for everything titled with a *blue* verb so we
    # at least mark the demo as "defensive observation" instead of empty.
    "detecting": ["T1059"],
    "detection": ["T1059"],
    # ---- round 4: more advanced offensive + tradecraft ---------------------
    "attacking corporate wi-fi": ["T1110", "T1040", "T1557"],
    "wi-fi password cracking": ["T1110", "T1040"],
    "windows low level detectability": ["T1027", "T1055", "T1562"],
    "windows api monitoring": ["T1106", "T1055"],   # Native API / Process Injection
    "windows api hooking": ["T1574", "T1055"],   # Hijack Execution Flow
    "wmi tradecraft": ["T1047"],   # Windows Management Instrumentation
    "persistence tradecraft": ["T1547", "T1543"],   # Boot/Logon Autostart / Create or Modify System Process
    "android attacks": ["T1418", "T1623"],
    # ---- AI / LLM (MITRE ATLAS overlap; ATT&CK Enterprise has limited
    # coverage so we use the closest enterprise IDs where applicable). The
    # bulk of these stay empty because the academy module IS the source of
    # the technique, not the consumer.
    "llm output attacks": [],
    "attacking ai": [],
    "ai evasion": [],
    "ai defense": [],
    "ai privacy": [],
    "fundamentals of ai": [],
    "applications of ai in infosec": [],
    "red teaming ai": [],
    # ---- additional process / foundational (intentional empties) -----------
    "introduction to c#": [],
    "security incident reporting": [],
    "intro to academy's purple modules": [],
    "purple modules": [],
    "introduction to penetration testing": ["T1595", "T1018", "T1046"],
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
