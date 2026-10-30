#!/usr/bin/env python3
"""Verify the published Table 3 and Table 5 cells against the camera-ready.

FoRGe cells use the per-task rollup the harness stored at generation time;
Claude cells are re-scored from the shipped regexes. Exclusions are per metric:
TPR drops single-positive tasks and the firewall task (n=55); FPR drops tasks
with no negatives in the evaluated universe and the firewall task (n=62).
See the "published aggregation" block in score_tables.py for why.

Reads only files inside this artifact. Stdlib only.

    python3 scripts/verify_published_cells.py [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_tables as st

st.require_results()

FW = st.FIREWALL
DP = {"tpr": 2, "fpr": 3}


TABLE1 = [("Number of tasks", 66), ("Total log lines", 19428), ("Total positives", 501),
          ("Total negatives", 18927), ("Average log lines per task", 294.36),
          ("Average positives per task", 7.59), ("Average negatives per task", 286.77)]
TABLE6 = [("Mean", 7.68), ("Median", 8), ("Max", 15), ("Min", 2)]


def verify_dataset_tables():
    import glob as _g, json as _j, os as _o, statistics as _s
    files = sorted(_g.glob(str(st.DATASET / "*.json")))
    tp = tn = 0
    lens, words = [], []
    for p in files:
        d = _j.load(open(p, encoding="utf-8"))
        a, b = len(d["true_positives"]), len(d["true_negatives"])
        tp += a; tn += b; lens.append(a + b); words.append(len(d["description"].split()))
    got = dict(zip([n for n, _ in TABLE1],
                   [len(files), tp + tn, tp, tn, round(_s.fmean(lens), 2),
                    round(tp / len(files), 2), round(tn / len(files), 2)]))
    srt = sorted(words)
    med = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
    got6 = dict(Mean=round(_s.fmean(words), 2), Median=med, Max=max(words), Min=min(words))
    ok = 0
    print("Table 1 -- dataset summary statistics")
    for name, pub in TABLE1:
        m = got[name]; good = round(float(m), 2) == round(float(pub), 2); ok += good
        print(f"  {name:<30} computed {m:>10}   paper {pub:>10}   {'MATCH' if good else 'MISMATCH'}")
    print("\nTable 6 -- description length in words, all 66 tasks")
    for name, pub in TABLE6:
        m = got6[name]; good = round(float(m), 2) == round(float(pub), 2); ok += good
        print(f"  {name:<8} computed {m:>8}   paper {pub:>8}   {'MATCH' if good else 'MISMATCH'}")
    print(f"\n{ok}/11 dataset-table values reproduce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--dataset", action="store_true",
                    help="verify Table 1 and Table 6 (dataset statistics) and exit")
    args = ap.parse_args()

    if args.dataset:
        verify_dataset_tables()
        return

    files, stems, data = st.load_dataset()
    splits = st.splits_by_run(files)
    single = st.single_positive_rules(data)
    EXCL = {"tpr": frozenset({FW}) | single, "fpr": frozenset({FW})}

    print("loading FoRGe stored per-task rollups ...", flush=True)
    forge = {c: st.forge_stored(c) for c in ("desc", "no_desc")}

    print("re-scoring Claude arms ...", flush=True)
    claude = {}
    claude[("T3", "desc")] = list(st.claude_retrospective(
        stems, data, "claude/with_description_retrospective",
        "claude_results_loop", "run_*", True))
    claude[("T3", "no_desc")] = list(st.claude_retrospective(
        stems, data, "claude/no_description_retrospective",
        "claude_results_no_description", "eval_*", True))
    cd = list(st.claude_continuous(stems, data, "claude/with_description_continuous",
                                   "claude_results_loop", "run_*", splits, True))
    cn = list(st.claude_continuous(stems, data, "claude/no_description_continuous",
                                   "claude_results_no_description", "eval_*", splits, True))
    claude[("T4", "desc")] = cd
    claude[("T4", "no_desc")] = cn

    rows = []
    for tag, side in (("T3", "holdout"), ("T5train", "train"), ("T5test", "test")):
        for agent in ("forge", "claude"):
            for cond in ("desc", "no_desc"):
                for metric in ("tpr", "fpr"):
                    key = (tag, agent, cond, metric)
                    pub, pci = st.PUBLISHED_CI[key]
                    if agent == "forge":
                        vals = forge[cond][("T3" if tag == "T3" else
                                            "T4train" if tag == "T5train" else "T4test", metric)]
                    else:
                        recs = claude[(("T3" if tag == "T3" else "T4"), cond)]
                        vals = st.records_to_values(recs, side, metric)
                    m, ci, n = st.agg_published(vals, EXCL[metric])
                    d = DP[metric]
                    match = m is not None and round(m, d) == round(pub, d)
                    ci_ok = (ci is not None
                             and round(ci, d) == round(pci, d))
                    rows.append(dict(cell=f"{tag} {agent} {cond} {metric.upper()}",
                                     published=pub, published_ci=pci,
                                     recomputed=m, recomputed_ci=ci, n=n,
                                     match=match, ci_match=ci_ok,
                                     within_published_ci=(m is not None and abs(m - pub) <= pci)))

    w = max(len(r["cell"]) for r in rows)
    print(f"\n{'cell':<{w}} {'published':>18} {'recomputed':>18} {'n':>4}  verdict")
    print("-" * (w + 52))
    nmatch = nci = 0
    for r in rows:
        d = DP["tpr"] if r["cell"].endswith("TPR") else DP["fpr"]
        rec = "   --" if r["recomputed"] is None else f"{r['recomputed']:.{d}f} +/- {r['recomputed_ci']:.{d}f}"
        v = "MATCH" if r["match"] else ("CHANGED, within published CI" if r["within_published_ci"]
                                        else "CHANGED, OUTSIDE published CI")
        if r["match"] and not r["ci_match"]:
            v += " (mean only; CI differs)"
        nmatch += r["match"]; nci += (r["match"] and r["ci_match"])
        print(f"{r['cell']:<{w}} {r['published']:>9.{d}f} +/- {r['published_ci']:<5.{d}f} {rec:>18} {r['n']:>4}  {v}")
    print("-" * (w + 52))
    print(f"{nmatch}/{len(rows)} means reproduce exactly; {nci}/{len(rows)} also match the published CI")
    outside = [r["cell"] for r in rows if not r["match"] and not r["within_published_ci"]]
    print(f"outside the published 95% CI: {outside if outside else 'none'}")

    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
