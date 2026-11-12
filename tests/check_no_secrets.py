#!/usr/bin/env python3
"""Fail if a file of the repository contains a credential shape, a secret file name, a network address or a login.

usage: python3 tests/check_no_secrets.py        (or: make secrets)

Standard library and git only. Scans what `git ls-files` lists (tracked and staged files); outside a git checkout
it scans every file of the tree except build directories. The patterns are shapes, not real tokens. Addresses and
logins are in the list because notes about private machines do not belong here either: they go to CLAUDE.local.md,
which git ignores.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2_000_000
SKIP_SUFFIX = {".gguf", ".bin", ".head", ".npy", ".npz", ".o", ".png", ".jpg", ".jpeg", ".gz", ".zip"}
SKIP_DIRS = {".git", "build", "models", "__pycache__"}
SECRET_NAMES = {".env", "credentials.json", "id_rsa", "id_ed25519", "hf.token"}
PATTERNS = {
    "Hugging Face token": r"hf_[A-Za-z0-9]{30,}",
    "GitHub token": r"(?:ghp|gho|ghu|ghs)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}",
    "GitLab token": r"glpat-[A-Za-z0-9_-]{20,}",
    "AWS key id": r"AKIA[0-9A-Z]{16}",
    "API key": r"sk-[A-Za-z0-9_-]{24,}",
    "private key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    # any IPv4 address except the loopback and the wildcard address
    "network address": r"(?<![\d.])(?!127\.0\.0\.1(?!\d)|0\.0\.0\.0(?!\d))(?:\d{1,3}\.){3}\d{1,3}(?![\d.])",
    # a name, an at sign and a host: a login on some machine or an e-mail address
    "login or e-mail address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*",
}
# shapes that look like a login but are not one
ALLOWED = {"noreply@anthropic.com"}


def files():
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, stderr=subprocess.DEVNULL)
        names = [raw.decode("utf-8") for raw in out.split(b"\0") if raw]
        if names:
            return names
    except (OSError, subprocess.CalledProcessError):
        pass
    found = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if path.is_file() and not any(part in SKIP_DIRS or part.startswith("build-") for part in rel.parts):
            found.append(rel.as_posix())
    return found


def main():
    compiled = {name: re.compile(pattern) for name, pattern in PATTERNS.items()}
    bad = []
    names = files()
    for rel in names:
        path = ROOT / rel
        if path.name in SECRET_NAMES or path.name.startswith(".env."):
            bad.append(f"{rel}: secret file name")
            continue
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIX or path.stat().st_size > MAX_BYTES:
            continue
        data = path.read_bytes()
        if b"\0" in data[:8000]:
            continue
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            for name, pattern in compiled.items():
                for match in pattern.finditer(line):
                    if match.group(0) not in ALLOWED:
                        bad.append(f"{rel}:{number}: {name}: {match.group(0)[:60]}")
    if bad:
        print("\n".join(bad))
        print(f"{len(bad)} finding(s) in {len(names)} files")
        return 1
    print(f"no credential shapes, addresses or logins in {len(names)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
