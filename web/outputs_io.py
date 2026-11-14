"""Read-only parsers for the committed outputs/ files and README sections."""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
README = os.path.join(ROOT, "README.md")


def _coerce(v):
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    return v


def read_results(name):
    """Parse 'key = value' lines into an ordered dict (numbers become int/float)."""
    res = {}
    with open(os.path.join(OUT_DIR, name), encoding="utf-8") as f:
        for line in f:
            if " = " not in line:
                continue
            k, v = line.rstrip("\n").split(" = ", 1)
            res[k] = _coerce(v)
    return res


def read_csv_table(name):
    with open(os.path.join(OUT_DIR, name), newline="", encoding="utf-8") as f:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]


def readme_sections():
    """Map '## Heading' -> body text of the root README."""
    with open(README, encoding="utf-8") as f:
        text = f.read()
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
