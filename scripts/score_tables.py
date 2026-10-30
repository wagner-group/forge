#!/usr/bin/env python3
"""Scoring library for the published Table 3 and Table 5 cells.

Uses ONLY files inside this artifact. Pure local regex matching -- no agent, no
API calls. Stdlib only.

Two regex conventions are preserved exactly as the two original scorers used
them. They differ, and each arm is scored under its own:

  * FoRGe / Gemini arms  -- re.IGNORECASE | re.DOTALL
    (splunk_rule_testing/detection/regex.py:78)
  * Claude arms          -- no flags
    (scripts/claude_scoring/score_eval.py:175)

Universes:
  retrospective  TPR over (true_positives - seed), FPR over all true_negatives
  continuous     train: (tp_train - seeds) / tn_train
                 test:  tp_test / tn_test
                 splits regenerated from dataset/final with master_seed 161 via
                 eval_test_split/_splitting.py; verified to reproduce every
                 shipped split manifest and every shipped agent pool.

Usage:
This module is imported by the verification scripts; it is not a report
generator. Run scripts/verify_published_cells.py and the other verifiers.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import signal
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "final"
RESULTS = ROOT / "results"


def require_results() -> None:
    """Exit cleanly when the separately-deposited results/ tree is absent.

    The experiment output is ~1.6 GB and is not committed to this repository, so
    every script that scores it must say so rather than fail with a traceback or
    silently report that nothing reproduces.
    """
    if RESULTS.is_dir() and any(RESULTS.iterdir()):
        return
    import sys as _sys
    print(
        "The experimental data is not present.\n"
        "\n"
        f"  expected: {RESULTS}\n"
        "\n"
        "It is deposited separately (about 195 MB compressed, 1.6 GB unpacked)\n"
        "because it is 20,000 files. Download it, unpack the per-arm archives as\n"
        "results/ at the root of this repository, and re-run this script.\n"
        "\n"
        'See README.md, section "Experimental data", for the deposit link.',
        file=_sys.stderr,
    )
    raise SystemExit(2)
FIREWALL = "linux_auditd_disable_or_modify_system_firewall"

# Scoring-time label override carried over verbatim from score_eval.py:30-35.
_FW_FORCE_POSITIVE = frozenset({
    'type=SERVICE_STOP msg=audit(1722965483.442:15704): pid=1 uid=0 auid=4294967295 ses=4294967295 subj=unconfined msg=\'unit=firewalld comm="systemd" exe="/usr/lib/systemd/systemd" hostname=? addr=? terminal=? res=success\'\x1dUID="root" AUID="unset"',
    'type=SERVICE_STOP msg=audit(1722965483.442:15704): pid=1 uid=0 auid=4294967295 ses=4294967295 subj=unconfined msg=\'unit=firewalld comm="systemd" exe="/usr/lib/systemd/systemd" hostname=? addr=? terminal=? res=success\'',
})

GEMINI_FLAGS = re.IGNORECASE | re.DOTALL
CLAUDE_FLAGS = 0


def load_dataset():
    files = sorted(glob.glob(str(DATASET / "*.json")))
    assert len(files) == 66, f"expected 66 tasks, found {len(files)}"
    stems, data = [], {}
    for p in files:
        stem = os.path.basename(p)[:-5]
        stems.append(stem)
        d = json.load(open(p, encoding="utf-8"))
        data[stem] = (list(d["true_positives"]), list(d["true_negatives"]))
    return files, stems, data


def slot_map(stems):
    return {f"{i:02d}": s for i, s in enumerate(stems, 1)}


_cache: dict[tuple[str, int], object] = {}


def compiled(pattern, flags):
    key = (pattern, flags)
    if key not in _cache:
        try:
            _cache[key] = re.compile(pattern, flags)
        except re.error:
            _cache[key] = None
    return _cache[key]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


# A small fraction of generated patterns backtrack catastrophically on 20 kB
# records. A 10 s cap bounds them, with one caveat worth knowing: Python runs
# signal handlers between bytecode instructions, so the cap fires between
# matches, not inside a single re.search() that never returns. It therefore
# bounds a slow *sequence* of matches but not one pathological match. The
# Gemini and Claude arms scored here contain no such case (0 and 4 trials hit
# the cap respectively, all between matches); scripts/verify_table4.py scores
# the Codex arm in a child process because that arm does contain them.
# Trials that hit the cap are counted and reported, never silently scored as 0.
PATTERN_TIMEOUT_S = 10.0
TIMEOUTS: list[tuple[str, str]] = []
signal.signal(signal.SIGALRM, _alarm)


def _count(pat, lines):
    n = 0
    for ln in lines:
        if pat.search(ln):
            n += 1
    return n


def score(pattern, flags, tps, tns, overrides=False, rule=""):
    """Return (tpr, fpr, tp_hits, fp_hits, n_tp, n_tn); tpr/fpr None if no
    universe or if the pattern exceeded PATTERN_TIMEOUT_S."""
    tps, tns = set(tps), set(tns)
    if overrides and rule == FIREWALL:
        tns -= _FW_FORCE_POSITIVE
        tps |= _FW_FORCE_POSITIVE
    pat = compiled(pattern, flags) if pattern else None
    if pat is None:
        return (0.0 if tps else None, 0.0 if tns else None, 0, 0, len(tps), len(tns))
    signal.setitimer(signal.ITIMER_REAL, PATTERN_TIMEOUT_S)
    try:
        tp_hits = _count(pat, tps)
        fp_hits = _count(pat, tns)
    except _Timeout:
        TIMEOUTS.append((rule, pattern[:80]))
        return (None, None, None, None, len(tps), len(tns))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return (tp_hits / len(tps) if tps else None,
            fp_hits / len(tns) if tns else None,
            tp_hits, fp_hits, len(tps), len(tns))


def splits_by_run(files):
    sys.path.insert(0, str(ROOT / "splunk_rule_testing"))
    from eval_test_split._splitting import load_rule_slots, build_splits_for_run, run_seed
    slots = load_rule_slots(files)
    out = {}
    for i in range(1, 11):
        by, man = build_splits_for_run(slots, run_seed(161, i), 0.5)
        out[i] = ({os.path.basename(k)[:-5]: v for k, v in by.items()}, man)
    return out


_bs_cache: dict = {}


def _batch_seeds(rule_split, run, n_cap=10):
    """compute_batch_seeds() for one (rule, run), memoised."""
    key = (rule_split.slot.file_name, run)
    if key not in _bs_cache:
        sys.path.insert(0, str(ROOT / "splunk_rule_testing"))
        from eval_test_split._splitting import compute_batch_seeds, run_seed
        _bs_cache[key] = compute_batch_seeds(rule_split, run_seed(161, run), n_cap)
    return _bs_cache[key]


# ---------------------------------------------------------------- arm loaders
# Each loader yields dicts: {rule, run, tpr, fpr, side} with side in
# {"holdout", "train", "test"}. One record per trial.

def gemini_retrospective(stems, data, condition, overrides):
    """FoRGe retrospective.

    `seed_index` enumerates the *sampled* seed list, not the task's
    true_positives. eval/regex.py:66-75 builds it as

        random.Random(derive_seed(master_seed, "sample", file_name))
              .sample(true_positives, min(len(true_positives), N))

    and the held-out universe is every true positive != that seed. Indexing
    true_positives directly picks the wrong seed whenever the sample reorders
    them, which inflates recomputed TPR (FPR is unaffected -- it never uses the
    seed). The sample is reproduced here instead.
    """
    import random
    sys.path.insert(0, str(ROOT / "splunk_rule_testing"))
    from eval._seeding import derive_seed
    arm = "regex" if condition == "desc" else "regex_no_des"
    for rep in range(1, 11):
        armdir = RESULTS / f"forge_gemini/retrospective/rep{rep}/{arm}"
        summ = json.load(open(armdir / "summary.json", encoding="utf-8"))
        master_seed = summ["master_seed"]
        for p in sorted(glob.glob(str(armdir / "files/*.json"))):
            rule = os.path.basename(p)[:-5]
            if rule not in data:
                continue
            tps, tns = data[rule]
            obj = json.load(open(p, encoding="utf-8"))
            n_seeds = (obj.get("metrics") or {}).get("num_seeds") or len(obj.get("per_seed_results") or [])
            k = min(len(tps), n_seeds) if n_seeds else len(tps)
            sample_seed = derive_seed(master_seed, "sample", f"{rule}.json")
            seeds_to_test = random.Random(sample_seed).sample(tps, k) if k else []
            for t in obj.get("trials", []):
                if t.get("status") != "Success":
                    continue
                si = t.get("seed_index")
                seed = seeds_to_test[si] if isinstance(si, int) and si < len(seeds_to_test) else None
                univ = [x for x in tps if x != seed]
                tpr, fpr, *_ = score(t.get("rule"), GEMINI_FLAGS, univ, tns, overrides, rule)
                yield dict(rule=rule, run=rep, tpr=tpr, fpr=fpr, side="holdout")


def gemini_continuous(stems, data, condition, splits, overrides):
    arm = "with_description" if condition == "desc" else "no_description"
    for p in sorted(glob.glob(str(RESULTS / f"forge_gemini/continuous/{arm}/files/*.json"))):
        rule = os.path.basename(p)[:-5]
        if rule not in data:
            continue
        for t in json.load(open(p, encoding="utf-8")).get("trials", []):
            if t.get("status") != "Success":
                continue
            run = int(t["partition_idx"]) + 1
            sp = splits[run][0][rule]
            # The train-side seed is the seed_idx-th entry of the per-rule
            # deterministic shuffle of tp_train (_splitting.compute_batch_seeds),
            # not simply tp_train[0].
            batch_seeds = _batch_seeds(sp, run)
            si = t.get("seed_idx")
            seed = batch_seeds[si] if isinstance(si, int) and si < len(batch_seeds) else None
            tr_tp = [x for x in sp.tp_train if x != seed]
            for side, tp_u, tn_u in (("train", tr_tp, sp.tn_train),
                                     ("test", sp.tp_test, sp.tn_test)):
                tpr, fpr, *_ = score(t.get("rule"), GEMINI_FLAGS, tp_u, tn_u, overrides, rule)
                yield dict(rule=rule, run=run, tpr=tpr, fpr=fpr, side=side)


def _claude_batches(run_dir, results_subdir):
    for bf in sorted(glob.glob(str(Path(run_dir) / results_subdir / "batch_*_result.json"))):
        d = json.load(open(bf, encoding="utf-8"))
        for e in d.get("results", []):
            yield e


def claude_retrospective(stems, data, tree, results_subdir, run_glob, overrides):
    smap = slot_map(stems)
    for run_dir in sorted(glob.glob(str(RESULTS / tree / run_glob))):
        run = int(re.search(r"(\d+)$", os.path.basename(run_dir.rstrip("/"))).group(1))
        for e in _claude_batches(run_dir, results_subdir):
            rule = smap.get(str(e.get("name")))
            if rule is None:
                continue
            tps, tns = data[rule]
            seed = e.get("true_positive_example")
            univ = [x for x in tps if x != seed]
            tpr, fpr, *_ = score(e.get("regex"), CLAUDE_FLAGS, univ, tns, overrides, rule)
            yield dict(rule=rule, run=run, tpr=tpr, fpr=fpr, side="holdout")


def claude_continuous(stems, data, tree, results_subdir, run_glob, splits, overrides):
    smap = slot_map(stems)
    for run_dir in sorted(glob.glob(str(RESULTS / tree / run_glob))):
        run = int(re.search(r"(\d+)$", os.path.basename(run_dir.rstrip("/"))).group(1))
        for e in _claude_batches(run_dir, results_subdir):
            rule = smap.get(str(e.get("name")))
            if rule is None:
                continue
            sp = splits[run][0][rule]
            seed = e.get("true_positive_example")
            tr_tp = [x for x in sp.tp_train if x != seed]
            for side, tp_u, tn_u in (("train", tr_tp, sp.tn_train),
                                     ("test", sp.tp_test, sp.tn_test)):
                tpr, fpr, *_ = score(e.get("regex"), CLAUDE_FLAGS, tp_u, tn_u, overrides, rule)
                yield dict(rule=rule, run=run, tpr=tpr, fpr=fpr, side=side)


def build_seed_source(stems):
    """(kind, run, batch, rule) -> seed log, from the corrected runs' seed dirs."""
    smap = slot_map(stems)
    out = {}
    for kind, tree, glob_pat in (("retrospective", "claude/with_description_retrospective", "run_*"),
                                 ("continuous", "claude/with_description_continuous", "run_*")):
        for run_dir in sorted(glob.glob(str(RESULTS / tree / glob_pat))):
            run = int(re.search(r"(\d+)$", os.path.basename(run_dir.rstrip("/"))).group(1))
            for sp in sorted(glob.glob(str(Path(run_dir) / "single_positive_log_with_description" / "batch_*" / "*.json"))):
                batch = os.path.basename(os.path.dirname(sp))
                rule = smap.get(os.path.basename(sp)[:-5])
                if rule is None:
                    continue
                obj = json.load(open(sp, encoding="utf-8"))
                out[(kind, run, batch, rule)] = obj.get("true_positive") or obj.get("true_positive_example")
    return out


# ------------------------------------------------------------- aggregation
def aggregate(records, side, metric, exclude=frozenset()):
    """Per-rule mean across runs, then macro-mean across rules (the convention
    the no-description cells reproduce exactly). Returns (value, n_rules, ci95)."""
    per_rule = defaultdict(list)
    for r in records:
        if r["side"] != side or r[metric] is None or r["rule"] in exclude:
            continue
        per_rule[r["rule"]].append(r[metric])
    rule_means = [statistics.fmean(v) for v in per_rule.values() if v]
    if not rule_means:
        return None, 0, None
    m = statistics.fmean(rule_means)
    ci = None
    if len(rule_means) > 1:
        sd = statistics.stdev(rule_means)
        ci = 1.96 * sd / math.sqrt(len(rule_means))
    return m, len(rule_means), ci


def single_positive_rules(data):
    return {r for r, (tps, _) in data.items() if len(tps) == 1}


def single_positive_rules_split(data, splits, side):
    """Rules whose universe on `side` has exactly one positive, in any run."""
    out = defaultdict(set)
    for run, (by, _) in splits.items():
        for rule, sp in by.items():
            n = len(sp.tp_test) if side == "test" else max(0, len(sp.tp_train) - 1)
            if n == 1:
                out[run].add(rule)
    return out


# ------------------------------------------------- published aggregation
# Two things differ from a naive re-scoring and both matter:
#
#   1. FoRGe values come from the per-task `metrics` rollup the harness wrote at
#      generation time, NOT from re-executing the regex. Re-executing agrees
#      with every stored PER-TRIAL metric (see check_internal_consistency.py)
#      but the harness's per-task rollup is not the mean of the per-trial rates,
#      so the cell values differ by 0.2-0.8 points.
#   2. Exclusions are per metric: TPR drops single-positive tasks AND the
#      firewall task (n=55); FPR drops only tasks with no negatives in the
#      evaluated universe AND the firewall task (n=62). Dropping
#      single-positive tasks from FPR as well gives n=52 and is wrong.
#
# The firewall exclusion is undocumented in the paper. Its labels were corrected
# on 2026-06-01, after some runs were generated, so its evaluated positive set
# differs between agents.

# Published values: Table 3 (retrospective) and Table 5
# (continuous monitoring). Several of these changed from the submission draft.
# Section 4 of the camera-ready documents the convention in full:
#   TPR = k/(n-1) per trial, seed excluded from the denominator; <=10 positives
#   sampled per task; per-task mean over positives; per-task rate averaged over
#   the 10 repetitions; macro-average across tasks; 95% CI = 1.96*s/sqrt(n)
#   across tasks. Excluded: 10 single-positive tasks and 1 late-stage relabeled
#   task for TPR (n=55); the same relabeled task and tasks with no negatives in
#   scope for FPR (n=62).
PUBLISHED_CI = {
    ("T3", "forge", "desc", "tpr"): (93.21, 3.38), ("T3", "forge", "desc", "fpr"): (0.120, 0.104),
    ("T3", "claude", "desc", "tpr"): (79.12, 6.29), ("T3", "claude", "desc", "fpr"): (0.056, 0.070),
    ("T3", "forge", "no_desc", "tpr"): (88.48, 4.72), ("T3", "forge", "no_desc", "fpr"): (0.186, 0.229),
    ("T3", "claude", "no_desc", "tpr"): (52.34, 9.35), ("T3", "claude", "no_desc", "fpr"): (0.090, 0.158),
    ("T5train", "forge", "desc", "tpr"): (95.21, 2.75), ("T5train", "forge", "desc", "fpr"): (0.119, 0.122),
    ("T5train", "claude", "desc", "tpr"): (81.26, 5.67), ("T5train", "claude", "desc", "fpr"): (0.084, 0.111),
    ("T5train", "forge", "no_desc", "tpr"): (87.42, 5.10), ("T5train", "forge", "no_desc", "fpr"): (0.138, 0.121),
    ("T5train", "claude", "no_desc", "tpr"): (52.21, 9.24), ("T5train", "claude", "no_desc", "fpr"): (0.348, 0.632),
    ("T5test", "forge", "desc", "tpr"): (92.91, 4.20), ("T5test", "forge", "desc", "fpr"): (0.125, 0.119),
    ("T5test", "claude", "desc", "tpr"): (78.93, 6.88), ("T5test", "claude", "desc", "fpr"): (0.084, 0.078),
    ("T5test", "forge", "no_desc", "tpr"): (85.76, 6.15), ("T5test", "forge", "no_desc", "fpr"): (0.399, 0.531),
    ("T5test", "claude", "no_desc", "tpr"): (51.94, 9.53), ("T5test", "claude", "no_desc", "fpr"): (0.342, 0.632),
}

# Table 4, the same-model control (both harnesses on GPT-5.6 Luna, retrospective).
PUBLISHED_T4 = {
    ("forge_luna", "desc", "tpr"): (89.23, 3.36), ("forge_luna", "desc", "fpr"): (0.184, 0.215),
    ("codex_luna", "desc", "tpr"): (70.14, 5.04), ("codex_luna", "desc", "fpr"): (7.635, 1.758),
    ("forge_luna", "no_desc", "tpr"): (77.01, 7.06), ("forge_luna", "no_desc", "fpr"): (1.867, 2.312),
    ("codex_luna", "no_desc", "tpr"): (42.64, 7.72), ("codex_luna", "no_desc", "fpr"): (3.441, 2.179),
}

# Section 5.1 in-text deltas, in TPR points (FoRGe-Luna minus Codex-Luna).
PUBLISHED_DELTAS = {
    ("retrospective", "desc"): 19.1, ("retrospective", "no_desc"): 34.4,
    ("continuous_test", "desc"): 14.7, ("continuous_test", "no_desc"): 25.7,
}

# Appendix A worked examples, both from FoRGe (Gemini) retrospective, Table 3 runs.
PUBLISHED_APPENDIX = [
    dict(example=1, task="dump_lsass_via_procdump", condition="no_desc",
         regex_len=26, held_out_tps=31, tp_hits=31, negatives=270, fp_hits=0),
    dict(example=2, task="powershell_loading_dotnet_into_memory_via_reflection",
         condition="desc", regex_len=55, held_out_tps=18, tp_hits=18,
         negatives=191, fp_hits=0),
]


def forge_stored(condition):
    """FoRGe per-task rates as the harness stored them. Returns
    {(tag, metric): {(task, rep): [values]}} for tag in T3/T4train/T4test."""
    out = defaultdict(lambda: defaultdict(list))
    arm = "regex" if condition == "desc" else "regex_no_des"
    for rep in range(1, 11):
        for p in sorted(glob.glob(str(RESULTS / f"forge_gemini/retrospective/rep{rep}/{arm}/files/*.json"))):
            task = os.path.basename(p)[:-5]
            j = json.load(open(p, encoding="utf-8"))
            m = j.get("metrics") or {}
            out[("T3", "tpr")][(task, rep)].append(m.get("recall"))
            tns = next((t.get("total_tns") for t in (j.get("trials") or []) if t.get("total_tns")), None)
            if m.get("avg_fp_hits") is not None and tns:
                out[("T3", "fpr")][(task, rep)].append(m["avg_fp_hits"] / tns)
    cdir = "with_description" if condition == "desc" else "no_description"
    for p in sorted(glob.glob(str(RESULTS / f"forge_gemini/continuous/{cdir}/files/*.json"))):
        task = os.path.basename(p)[:-5]
        j = json.load(open(p, encoding="utf-8"))
        m = j.get("metrics") or {}
        out[("T4train", "tpr")][(task, "all")] = [m.get("recall")]
        ok = [t["fpr"] for t in j.get("trials", [])
              if t.get("status") == "Success" and t.get("fpr") is not None]
        if ok:
            out[("T4train", "fpr")][(task, "all")] = ok
        out[("T4test", "tpr")][(task, "all")] = [m.get("average_test_true_positive_rate")]
        out[("T4test", "fpr")][(task, "all")] = [m.get("average_test_false_positive_rate")]
    return out


def agg_published(values, exclude):
    """values: {(task, rep): [rates]} -> (mean%, ci95 half-width%, n).
    Per-task mean over reps, then unweighted macro-mean over tasks; CI across
    tasks with z = 1.96."""
    per_task = defaultdict(list)
    for (task, rep), xs in values.items():
        xs = [x for x in xs if x is not None]
        if xs and task not in exclude:
            per_task[task].append(statistics.fmean(xs))
    task_means = [statistics.fmean(v) for v in per_task.values() if v]
    if not task_means:
        return None, None, 0
    m = statistics.fmean(task_means) * 100
    ci = None
    if len(task_means) > 1:
        ci = 1.96 * statistics.stdev(task_means) / math.sqrt(len(task_means)) * 100
    return m, ci, len(task_means)


def records_to_values(records, side, metric):
    out = defaultdict(list)
    for r in records:
        if r["side"] == side:
            out[(r["rule"], r["run"])].append(r[metric])
    return out


def fmt(v, scale=100, nd=2):
    return "   --  " if v is None else f"{v*scale:{6}.{nd}f}"


def main():
    print(__doc__.strip())
    print("\nThis module is the scoring library. For the published table values run:")
    print("    python3 scripts/verify_published_cells.py")
    print("    python3 scripts/check_internal_consistency.py")
    print("    python3 scripts/run10_sensitivity.py")


if __name__ == "__main__":
    main()
