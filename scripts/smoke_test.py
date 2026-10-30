#!/usr/bin/env python3
"""Fast check (well under a second) that this artifact is intact and scores correctly.

Verifies structure, then re-scores one task end to end and checks the result
against a figure published in the paper. Stdlib only, no network, no API key.

    python3 scripts/smoke_test.py

Exits 0 if every check passes. For the full verification run:
    scripts/verify_published_cells.py, verify_table4.py, verify_appendix.py
"""
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / "results").is_dir() or not any((ROOT / "results").iterdir()):
    print(
        "The experimental data is not present.\n"
        "\n"
        f'  expected: {ROOT / "results"}\n'
        "\n"
        "It is deposited separately (about 195 MB compressed, 1.6 GB unpacked).\n"
        "Download it, unpack the per-arm archives as results/ at the root of this\n"
        "repository, and re-run this script.\n"
        "\n"
        'See README.md, section "Experimental data", for the deposit link.',
        file=sys.stderr,
    )
    raise SystemExit(2)
fails = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(label)


print("structure")
tasks = sorted(glob.glob(str(ROOT / "dataset/final/*.json")))
check("dataset/final holds 66 tasks", len(tasks) == 66, f"found {len(tasks)}")
_mapping = ROOT / "dataset/filename_mapping.md"
if _mapping.exists():
    check("filename_mapping.md present", True)
else:
    print("  [skip] dataset/filename_mapping.md not generated yet "
          "(scripts/make_filename_mapping.py --write)")
for d, label in [("results/forge_gemini/retrospective", "FoRGe-Gemini retrospective"),
                 ("results/forge_gemini/continuous", "FoRGe-Gemini continuous"),
                 ("results/claude/with_description_retrospective", "Claude with-desc retrospective"),
                 ("results/claude/no_description_retrospective", "Claude no-desc retrospective"),
                 ("results/forge_luna/retrospective", "FoRGe-Luna retrospective"),
                 ("results/codex_luna", "Codex-Luna")]:
    check(f"{label} present", (ROOT / d).is_dir())
for f in ["splunk_rule_testing/agent.py", "splunk_rule_testing/claude/agent.py",
          "splunk_rule_testing/codex_agent.py", "splunk_rule_testing/llm_openai.py",
          "scripts/verify_published_cells.py", "LICENSE", "README.md"]:
    check(f"{f} present", (ROOT / f).exists())

print("\ndataset statistics (Table 1)")
tp = sum(len(json.load(open(p, encoding="utf-8"))["true_positives"]) for p in tasks)
tn = sum(len(json.load(open(p, encoding="utf-8"))["true_negatives"]) for p in tasks)
check("19,428 log lines", tp + tn == 19428, f"{tp + tn}")
check("501 positives / 18,927 negatives", (tp, tn) == (501, 18927), f"{tp} / {tn}")

print("\nsplit regeneration (master_seed 161)")
sys.path.insert(0, str(ROOT / "splunk_rule_testing"))
from eval_test_split._splitting import load_rule_slots, build_splits_for_run, run_seed  # noqa: E402
_, man = build_splits_for_run(load_rule_slots(tasks), run_seed(161, 1), 0.5)
check("run_01 manifest matches the shipped one",
      (man["train_lines"], man["test_lines"], man["total_lines"]) == (9747, 9681, 19428),
      f"{man['train_lines']}/{man['test_lines']}/{man['total_lines']}")

print("\nend-to-end scoring: Appendix A example 1 (dump_lsass_via_procdump)")
from eval._seeding import derive_seed  # noqa: E402
task = "dump_lsass_via_procdump"
d = json.load(open(ROOT / f"dataset/final/{task}.json", encoding="utf-8"))
tps, tns = d["true_positives"], d["true_negatives"]
ad = ROOT / "results/forge_gemini/retrospective/rep1/regex_no_des"
master = json.load(open(ad / "summary.json", encoding="utf-8"))["master_seed"]
obj = json.load(open(ad / f"files/{task}.json", encoding="utf-8"))
k = min(len(tps), (obj.get("metrics") or {}).get("num_seeds") or 0)
seeds = random.Random(derive_seed(master, "sample", f"{task}.json")).sample(tps, k)
found = None
for t in obj.get("trials", []):
    if t.get("status") != "Success" or t.get("seed_index") != 4:
        continue
    seed = seeds[t["seed_index"]]
    univ = [x for x in tps if x != seed]
    pat = re.compile(t["rule"], re.IGNORECASE | re.DOTALL)
    found = (len(t["rule"]), len(univ), sum(1 for l in univ if pat.search(l)),
             len(tns), sum(1 for l in tns if pat.search(l)))
    break
check("26-char regex, 31/31 held-out positives, 0/270 negatives",
      found == (26, 31, 31, 270, 0), str(found))

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {', '.join(fails)}")
    sys.exit(1)
print("all checks passed")
