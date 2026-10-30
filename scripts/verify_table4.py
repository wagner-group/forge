#!/usr/bin/env python3
"""Verify Table 4 (the GPT-5.6 Luna same-model control) and the Section 5.1 deltas.

Table 4 is retrospective only. Section 5.1 also quotes four TPR-point deltas
between FoRGe-Luna and Codex-Luna, two retrospective and two continuous-test.

FoRGe-Luna values come from the per-task rollup the harness stored, as for
FoRGe/Gemini. Codex-Luna rules are re-executed as for the Claude agent: Python
`re`, no flags, unanchored search over raw lines, seed removed from the TPR
universe. Continuous universes come from splits regenerated with master_seed 161.

Each Codex trial is scored in a child process with a 10 s wall-clock cap, since
a single catastrophically backtracking `re.search` cannot be interrupted
in-process. A pattern that exceeds the cap is scored as matching nothing, which
is the policy that reproduces the published figures; `--timeout-policy exclude`
drops those trials instead.

Reads only files inside this artifact. Stdlib only.
"""
import argparse
import glob
import json
import os
import re
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_tables as st  # noqa: E402

st.require_results()

CLAUDE_FLAGS = 0
TIMEOUT = 120.0  # overridable with --timeout; see the note in main()


# A single re.search() that backtracks for minutes cannot be interrupted
# in-process: Python runs signal handlers between bytecode instructions, and the
# match never yields. Codex patterns include such cases, so each Codex trial is
# scored in a child process with a wall-clock timeout.
def _worker(rx, tp_u, tn_u, q):
    try:
        pat = re.compile(rx) if rx else None
    except re.error:
        q.put(("bad", None, None))
        return
    if pat is None:
        q.put(("bad", None, None))
        return
    q.put(("ok",
           sum(1 for l in tp_u if pat.search(l)),
           sum(1 for l in tn_u if pat.search(l))))


def _score(rx, tp_u, tn_u, policy):
    """Returns (tpr, fpr) or ("SKIP", "SKIP") when the trial is dropped."""
    tp_u, tn_u = list(tp_u), list(tn_u)
    q = mp.Queue()
    proc = mp.Process(target=_worker, args=(rx, tp_u, tn_u, q), daemon=True)
    proc.start()
    proc.join(TIMEOUT)
    if proc.is_alive():
        proc.terminate(); proc.join()
        _score.timeouts += 1
        if policy == "exclude":
            return "SKIP", "SKIP"
        t = f = 0                      # scored as matching nothing
    else:
        try:
            status, t, f = q.get_nowait()
        except Exception:
            status, t, f = "bad", None, None
        if status == "bad":
            return (0.0 if tp_u else None), (0.0 if tn_u else None)
    return (t / len(tp_u) if tp_u else None), (f / len(tn_u) if tn_u else None)


_score.timeouts = 0


def forge_luna():
    """{(mode, cond, metric): {(task, rep): [rates]}} from the stored rollups."""
    out = defaultdict(lambda: defaultdict(list))
    for cond in ("desc", "no_desc"):
        for rep in range(1, 11):
            d = ROOT / f"results/forge_luna/retrospective/{cond}_rep{rep:02d}/files"
            for p in sorted(glob.glob(str(d / "*.json"))):
                task = os.path.basename(p)[:-5]
                j = json.load(open(p, encoding="utf-8"))
                m = j.get("metrics") or {}
                out[("retro", cond, "tpr")][(task, rep)].append(m.get("recall"))
                tns = next((t.get("total_tns") for t in (j.get("trials") or []) if t.get("total_tns")), None)
                if m.get("avg_fp_hits") is not None and tns:
                    out[("retro", cond, "fpr")][(task, rep)].append(m["avg_fp_hits"] / tns)
        cd = "with_description" if cond == "desc" else "no_description"
        for p in sorted(glob.glob(str(ROOT / f"results/forge_luna/continuous/{cd}/files/*.json"))):
            task = os.path.basename(p)[:-5]
            m = (json.load(open(p, encoding="utf-8")).get("metrics") or {})
            out[("cont_test", cond, "tpr")][(task, "all")] = [m.get("average_test_true_positive_rate")]
            out[("cont_test", cond, "fpr")][(task, "all")] = [m.get("average_test_false_positive_rate")]
    return out


def codex_luna(stems, data, splits, policy):
    out = defaultdict(lambda: defaultdict(list))
    smap = {f"{i:02d}": s for i, s in enumerate(stems, 1)}

    # retrospective: regexes from the bundle, seeds from the shipped seed dirs
    bundle = json.load(open(ROOT / "results/codex_luna/retrospective_results_bundle.json", encoding="utf-8"))
    seeds = {}
    for v in ("with_description", "no_description"):
        for sp in glob.glob(str(ROOT / f"results/codex_luna/retrospective/{v}/run_*/*/batch_*/*.json")):
            parts = Path(sp).parts
            run = int(re.search(r"run_(\d+)", sp).group(1))
            batch = int(re.search(r"batch_(\d+)", parts[-2]).group(1))
            slot = os.path.basename(sp)[:-5]
            obj = json.load(open(sp, encoding="utf-8"))
            seeds[(v, run, batch, slot)] = obj.get("true_positive") or obj.get("true_positive_example")
    for e in bundle["entries"]:
        if e.get("mode") != "retrospective":
            continue
        cond = "desc" if e["variant"] == "with_description" else "no_desc"
        task = smap.get(str(e["name"]))
        if task is None:
            continue
        seed = seeds.get((e["variant"], e["run"], e["batch"], str(e["name"])))
        tps, tns = data[task]
        tp_u = [x for x in tps if x != seed]
        t, f = _score(e.get("regex"), tp_u, tns, policy)
        if t == "SKIP":
            continue
        out[("retro", cond, "tpr")][(task, e["run"])].append(t)
        out[("retro", cond, "fpr")][(task, e["run"])].append(f)

    # continuous test side: batch results + regenerated splits
    for v, cond in (("with_description", "desc"), ("no_description", "no_desc")):
        for bf in sorted(glob.glob(str(ROOT / f"results/codex_luna/continuous/{v}/run_*/*/batch_*_result.json"))):
            run = int(re.search(r"run_(\d+)", bf).group(1))
            for e in json.load(open(bf, encoding="utf-8")).get("results", []):
                task = smap.get(str(e.get("name")))
                if task is None:
                    continue
                sp = splits[run][0][task]
                t, f = _score(e.get("regex"), sp.tp_test, sp.tn_test, policy)
                if t == "SKIP":
                    continue
                out[("cont_test", cond, "tpr")][(task, run)].append(t)
                out[("cont_test", cond, "fpr")][(task, run)].append(f)
    return out, _score.timeouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-policy", choices=["no-match", "exclude"], default="no-match")
    ap.add_argument("--timeout", type=float, default=TIMEOUT,
                    help="per-trial wall-clock cap in seconds (default 120)")
    a = ap.parse_args()
    globals()["TIMEOUT"] = a.timeout

    files, stems, data = st.load_dataset()
    splits = st.splits_by_run(files)
    single = st.single_positive_rules(data)
    EXCL = {"tpr": frozenset({st.FIREWALL}) | single, "fpr": frozenset({st.FIREWALL})}

    print(f"timeout policy: {a.timeout_policy}, cap {TIMEOUT:g}s", flush=True)
    print("loading FoRGe-Luna rollups ...", flush=True)
    fl = forge_luna()
    print("re-scoring Codex-Luna ...", flush=True)
    cl, timeouts = codex_luna(stems, data, splits, a.timeout_policy)
    print(f"Codex pattern timeouts at {TIMEOUT:g}s: {timeouts}\n")

    rows, ok = [], 0
    print(f"{'Table 4 cell':<34} {'published':>18} {'recomputed':>18}  verdict")
    print("-" * 82)
    for cond in ("desc", "no_desc"):
        for who, src in (("forge_luna", fl), ("codex_luna", cl)):
            for metric in ("tpr", "fpr"):
                pub, pci = st.PUBLISHED_T4[(who, cond, metric)]
                m, ci, n = st.agg_published(src[("retro", cond, metric)], EXCL[metric])
                d = 2 if metric == "tpr" else 3
                good = m is not None and round(m, d) == round(pub, d)
                ok += good
                rec = "   --" if m is None else f"{m:.{d}f} +/- {ci:.{d}f}"
                print(f"{who+' '+cond+' '+metric.upper():<34} {pub:>9.{d}f} +/- {pci:<5.{d}f} {rec:>18}  "
                      f"{'MATCH' if good else 'MISMATCH'}")
    print("-" * 82)
    print(f"{ok}/8 Table 4 values reproduce exactly\n")

    print("Section 5.1 deltas (FoRGe-Luna TPR minus Codex-Luna TPR, points)")
    dok = 0
    for mode, label in (("retro", "retrospective"), ("cont_test", "continuous_test")):
        for cond in ("desc", "no_desc"):
            pub = st.PUBLISHED_DELTAS[(label, cond)]
            f_m, _, _ = st.agg_published(fl[(mode, cond, "tpr")], EXCL["tpr"])
            c_m, _, _ = st.agg_published(cl[(mode, cond, "tpr")], EXCL["tpr"])
            if f_m is None or c_m is None:
                print(f"  {label:<16} {cond:<8} -- insufficient data")
                continue
            delta = f_m - c_m
            good = round(delta, 1) == round(pub, 1)
            dok += good
            print(f"  {label:<16} {cond:<8} paper {pub:>5.1f}   recomputed {delta:>6.2f}   "
                  f"{'MATCH' if good else 'MISMATCH'}")
    print(f"\n{dok}/4 deltas reproduce at the published precision")
    sys.exit(0 if ok == 8 and dok == 4 else 1)


if __name__ == "__main__":
    main()
