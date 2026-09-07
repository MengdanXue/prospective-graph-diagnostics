"""Fail CI on broken LaTeX references, duplicate labels, or overflowing boxes."""
import argparse
from pathlib import Path
import re

PATTERNS = (
    r"Overfull \\[hv]box",
    r"(?:Reference|Citation) .+ undefined",
    r"There were undefined (?:references|citations)",
    r"Label .+ multiply defined",
    r"There were multiply-defined labels",
    r"destination with the same identifier",
    r"^! ",
)


def issues(text):
    return [line.strip() for line in text.splitlines()
            if any(re.search(pattern, line, re.I) for pattern in PATTERNS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise SystemExit("LaTeX log is empty")
    found = issues(text)
    if found:
        raise SystemExit("\n".join(found))
    print("LaTeX log: no undefined references, duplicate labels, or overfull boxes")


if __name__ == "__main__":
    main()
