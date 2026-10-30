#!/usr/bin/env python3
"""Generate dataset/filename_mapping.md, or verify the committed copy.

Slot NN is the NNth entry of sorted(dataset/final/*.json).

    python3 scripts/make_filename_mapping.py           # print to stdout
    python3 scripts/make_filename_mapping.py --check    # diff against the committed file
    python3 scripts/make_filename_mapping.py --write    # rewrite the committed file

Also checks the mapping against the shipped Claude batch results: for every
trial, the slot's rule should be the task whose true_positives contain that
trial's true_positive_example verbatim.

Stdlib only.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = [
    "# filename_mapping -- slot number -> rule stem",
    "#",
    "# Positional: slot NN is the NNth entry of sorted(dataset/final/*.json).",
    "# Verify against dataset/final with:",
    "#",
    "#     python3 scripts/make_filename_mapping.py --check",
    "#",
    "# Regenerate this file with --write instead of --check.",
    "",
    "| file | rule |",
    "|---|---|",
]


def stems():
    return [os.path.basename(p)[:-5] for p in sorted(glob.glob(str(ROOT / "dataset/final/*.json")))]


def render(ss):
    return "\n".join(HEADER + [f"| `{i:02d}.json` | `{s}` |" for i, s in enumerate(ss, 1)]) + "\n"


def cross_check(ss):
    """Verify the positional mapping against every shipped Claude batch result."""
    tps = {}
    for p in sorted(glob.glob(str(ROOT / "dataset/final/*.json"))):
        tps[os.path.basename(p)[:-5]] = set(json.load(open(p, encoding="utf-8"))["true_positives"])
    smap = {f"{i:02d}": s for i, s in enumerate(ss, 1)}
    agree = disagree = missing = 0
    for bf in glob.glob(str(ROOT / "results/claude/**/batch_*_result.json"), recursive=True):
        for e in json.load(open(bf, encoding="utf-8")).get("results", []):
            rule = smap.get(str(e.get("name")))
            seed = e.get("true_positive_example")
            if rule is None or seed is None:
                missing += 1
            elif seed in tps.get(rule, ()):
                agree += 1
            else:
                disagree += 1
    return agree, disagree, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    ss = stems()
    text = render(ss)
    target = ROOT / "dataset/filename_mapping.md"
    if a.check and not target.exists():
        print("dataset/filename_mapping.md does not exist yet. It is fully derived "
              "from dataset/final/; generate it with --write, then re-run --check.")
        raise SystemExit(2)
    if a.write:
        target.write_text(text)
        print(f"wrote {target}")
    elif a.check:
        on_disk = target.read_text()
        same = on_disk == text
        print(f"{len(ss)} tasks; committed file matches regeneration: {same}")
        agree, disagree, missing = cross_check(ss)
        total = agree + disagree
        print(f"cross-check against shipped Claude batch results: "
              f"{agree}/{total} trials agree, {disagree} disagree, {missing} unresolved")
        sys.exit(0 if same and disagree == 0 else 1)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
