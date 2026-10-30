#!/usr/bin/env python3
"""Re-execute every shipped FoRGe regex and compare against its stored metric.

For each trial, the stored regex is run against the universe that trial used and
the result compared with the metric stored beside it. This checks the artifact
against itself, independently of the paper: a disagreement means the shipped
dataset is not the one the stored metric was computed against.

Seed reconstruction matters here and is easy to get wrong:

  retrospective  seed_index enumerates the SAMPLED seed list, not the task's
                 true_positives -- eval/regex.py:66-75 builds it as
                 random.Random(derive_seed(master_seed, "sample", file_name))
                     .sample(true_positives, min(len(true_positives), N))
  continuous     seed_idx indexes compute_batch_seeds(), a per-rule
                 deterministic shuffle of tp_train -- not tp_train[0]

Indexing true_positives directly picks the wrong seed whenever the sample
reorders them, which leaves the wrong positive in the held-out universe and
inflates recomputed TPR while leaving FPR untouched.

Reads only files inside this artifact. Stdlib only.

    python3 scripts/check_internal_consistency.py
"""
import glob
import json
import os
import random
import re
import signal
import sys
from collections import Counter
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
sys.path.insert(0, str(ROOT / "splunk_rule_testing"))
from eval._seeding import derive_seed                                  # noqa: E402
from eval_test_split._splitting import (load_rule_slots,               # noqa: E402
                                        build_splits_for_run,
                                        compute_batch_seeds, run_seed)

GEM = re.IGNORECASE | re.DOTALL
TIMEOUT = 10.0


class _TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))


def rate(pat, lines):
    return (sum(1 for ln in lines if pat.search(ln)) / len(lines)) if lines else None


def cmp(tally, key, stored, mine):
    if stored is None and mine is None:
        tally[key + ":agree"] += 1
    elif stored is None or mine is None:
        tally[key + ":DISAGREE"] += 1
    elif abs(stored - mine) < 1e-9:
        tally[key + ":agree"] += 1
    else:
        tally[key + ":DISAGREE"] += 1


def report(label, tally, timeouts, skipped):
    print(f"\n{label}   timeouts={timeouts}  uncompilable={skipped}")
    keys = sorted({k.split(":")[0] for k in tally})
    for k in keys:
        a, d = tally[k + ":agree"], tally[k + ":DISAGREE"]
        n = a + d
        flag = "" if d == 0 else "   <-- MISMATCH"
        print(f"   {k:<9} agree {a:>5} / {n:<5} = {100*a/n if n else 0:6.2f}%{flag}")


def main():
    files = sorted(glob.glob(str(ROOT / "dataset/final/*.json")))
    data = {}
    for p in files:
        d = json.load(open(p, encoding="utf-8"))
        data[os.path.basename(p)[:-5]] = (d["true_positives"], d["true_negatives"])

    # ---- retrospective
    for arm, label in (("regex", "with description"), ("regex_no_des", "no description")):
        tally, to, skip = Counter(), 0, 0
        for rep in range(1, 11):
            armdir = ROOT / f"results/forge_gemini/retrospective/rep{rep}/{arm}"
            master = json.load(open(armdir / "summary.json", encoding="utf-8"))["master_seed"]
            for p in sorted(glob.glob(str(armdir / "files/*.json"))):
                rule = os.path.basename(p)[:-5]
                if rule not in data:
                    continue
                tps, tns = data[rule]
                obj = json.load(open(p, encoding="utf-8"))
                n_seeds = (obj.get("metrics") or {}).get("num_seeds") \
                    or len(obj.get("per_seed_results") or [])
                k = min(len(tps), n_seeds) if n_seeds else len(tps)
                seeds = random.Random(derive_seed(master, "sample", f"{rule}.json")).sample(tps, k) if k else []
                for t in obj.get("trials", []):
                    if t.get("status") != "Success":
                        continue
                    si = t.get("seed_index")
                    seed = seeds[si] if isinstance(si, int) and si < len(seeds) else None
                    univ = [x for x in tps if x != seed]
                    try:
                        pat = re.compile(t["rule"], GEM)
                    except re.error:
                        skip += 1
                        continue
                    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
                    try:
                        mine = rate(pat, univ)
                        mine = None if mine is None else min(mine, 1.0)
                    except _TO:
                        to += 1
                        continue
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    cmp(tally, "recall", t.get("recall"), mine)
        report(f"FoRGe retrospective, {label}:", tally, to, skip)

    # ---- continuous
    slots = load_rule_slots(files)
    splits = {}
    for i in range(1, 11):
        by, _ = build_splits_for_run(slots, run_seed(161, i), 0.5)
        splits[i] = {os.path.basename(k)[:-5]: v for k, v in by.items()}

    for arm, label in (("with_description", "with description"),
                       ("no_description", "no description")):
        tally, to, skip = Counter(), 0, 0
        for p in sorted(glob.glob(str(ROOT / f"results/forge_gemini/continuous/{arm}/files/*.json"))):
            rule = os.path.basename(p)[:-5]
            if rule not in data:
                continue
            for t in json.load(open(p, encoding="utf-8")).get("trials", []):
                if t.get("status") != "Success":
                    continue
                run = int(t["partition_idx"]) + 1
                sp = splits[run][rule]
                bs = compute_batch_seeds(sp, run_seed(161, run), 10)
                si = t.get("seed_idx")
                seed = bs[si] if isinstance(si, int) and si < len(bs) else None
                train_tp = [x for x in sp.tp_train if x != seed]
                try:
                    pat = re.compile(t["rule"], GEM)
                except re.error:
                    skip += 1
                    continue
                signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
                try:
                    vals = {"tpr": rate(pat, train_tp), "fpr": rate(pat, sp.tn_train),
                            "test_tpr": rate(pat, sp.tp_test), "test_fpr": rate(pat, sp.tn_test)}
                except _TO:
                    to += 1
                    continue
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                for k in vals:
                    cmp(tally, k, t.get(k), vals[k])
        report(f"FoRGe continuous, {label}:", tally, to, skip)

    print("\nAny MISMATCH above means the shipped dataset is not the state the stored")
    print("metric was computed against. All zero means the artifact is self-consistent")
    print("and any gap against the paper is an aggregation-convention difference.")


if __name__ == "__main__":
    main()
