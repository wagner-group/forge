#!/usr/bin/env python3
"""Table 5 with-description sensitivity to run_10, which is missing batch_007.

results/claude/with_description_continuous/run_10/ holds 9 of 10 output batches
(batch_007 is absent; its 7 seed logs were staged, so the gap is a missing agent
output, not a missing input). This reports run_10's own scores and whether
including or excluding the run moves any Table 5 with-description cell at the
precision the paper reports (2 dp for TPR, 3 dp for FPR).

Reads only files inside this artifact. Stdlib only.
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_tables as st

st.require_results()

FW = st.FIREWALL


def main():
    files, stems, data = st.load_dataset()
    splits = st.splits_by_run(files)
    recs = list(st.claude_continuous(stems, data,
                                    "claude/with_description_continuous",
                                    "claude_results_loop", "run_*", splits, True))
    print(f"trials scored: {len(recs)}   pattern timeouts: {len(st.TIMEOUTS)}")

    print("\nbatches present per run:")
    for i in range(1, 11):
        n = len(glob.glob(str(st.RESULTS / f"claude/with_description_continuous/run_{i:02d}"
                                           "/claude_results_loop/batch_*_result.json")))
        trials = sum(1 for r in recs if r["run"] == i and r["side"] == "test")
        print(f"  run_{i:02d}: {n:>2} batches, {trials:>3} test-side trials"
              + ("   <-- batch_007 missing" if n != 10 else ""))

    single = st.single_positive_rules(data)
    EXCL = {"tpr": frozenset({FW}) | single, "fpr": frozenset({FW})}

    print("\nrun_10 alone (published convention, n=55 TPR / 62 FPR):")
    only10 = [r for r in recs if r["run"] == 10]
    for side in ("train", "test"):
        t, tci, ntp = st.agg_published(st.records_to_values(only10, side, "tpr"), EXCL["tpr"])
        f, fci, nfp = st.agg_published(st.records_to_values(only10, side, "fpr"), EXCL["fpr"])
        print(f"  {side:<5} TPR {t:6.2f}% (n={ntp})   FPR {f:6.3f}% (n={nfp})")

    without = [r for r in recs if r["run"] != 10]
    print("\nTable 5 with-description cells, including vs excluding run_10")
    print(f"  {'cell':<12} {'incl. run_10':>14} {'excl. run_10':>14} {'delta':>10} {'moves at paper precision?':>28}")
    for side, metric, nd in (("train", "tpr", 2), ("train", "fpr", 3),
                             ("test", "tpr", 2), ("test", "fpr", 3)):
        a, _, _ = st.agg_published(st.records_to_values(recs, side, metric), EXCL[metric])
        b, _, _ = st.agg_published(st.records_to_values(without, side, metric), EXCL[metric])
        if a is None or b is None:
            print(f"  {side+' '+metric:<12} {'--':>14} {'--':>14}")
            continue
        a100, b100 = a, b          # agg_published already returns percent
        moved = round(a100, nd) != round(b100, nd)
        print(f"  {side+' '+metric:<12} {a100:>13.{nd}f}% {b100:>13.{nd}f}% "
              f"{a100-b100:>+9.{nd+2}f} {('YES' if moved else 'no'):>28}")


if __name__ == "__main__":
    main()
