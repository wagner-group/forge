#!/usr/bin/env python3
"""Verify the two worked examples in Appendix A of the paper.

Each example states five figures: regex length, held-out positive count,
positives matched, negative count, negatives matched. Both examples come from
the FoRGe (Gemini) retrospective runs reported in Table 3. This searches those
runs for a trial satisfying all five.

Reads only files inside this artifact. Stdlib only.
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
sys.path.insert(0, str(ROOT / "splunk_rule_testing"))
from eval._seeding import derive_seed  # noqa: E402

GEM = re.IGNORECASE | re.DOTALL
EXAMPLES = [
    dict(n=1, task="dump_lsass_via_procdump", arm="regex_no_des",
         regex_len=26, held_out=31, tp_hits=31, negatives=270, fp_hits=0),
    dict(n=2, task="powershell_loading_dotnet_into_memory_via_reflection", arm="regex",
         regex_len=55, held_out=18, tp_hits=18, negatives=191, fp_hits=0),
]


def main():
    ok = 0
    for ex in EXAMPLES:
        d = json.load(open(ROOT / f"dataset/final/{ex['task']}.json", encoding="utf-8"))
        tps, tns = d["true_positives"], d["true_negatives"]
        hit = None
        for rep in range(1, 11):
            ad = ROOT / f"results/forge_gemini/retrospective/rep{rep}/{ex['arm']}"
            sp = ad / f"files/{ex['task']}.json"
            if not sp.exists():
                continue
            master = json.load(open(ad / "summary.json", encoding="utf-8"))["master_seed"]
            obj = json.load(open(sp, encoding="utf-8"))
            k = min(len(tps), (obj.get("metrics") or {}).get("num_seeds") or 0)
            seeds = random.Random(derive_seed(master, "sample", f"{ex['task']}.json")).sample(tps, k) if k else []
            for t in obj.get("trials", []):
                if t.get("status") != "Success":
                    continue
                si = t.get("seed_index")
                seed = seeds[si] if isinstance(si, int) and si < len(seeds) else None
                univ = [x for x in tps if x != seed]
                pat = re.compile(t["rule"], GEM)
                tp_hits = sum(1 for l in univ if pat.search(l))
                fp_hits = sum(1 for l in tns if pat.search(l))
                if (len(t["rule"]) == ex["regex_len"] and len(univ) == ex["held_out"]
                        and tp_hits == ex["tp_hits"] and len(tns) == ex["negatives"]
                        and fp_hits == ex["fp_hits"]):
                    hit = (rep, si, t["rule"])
                    break
            if hit:
                break
        print(f"Example {ex['n']}: {ex['task']}")
        print(f"  paper: {ex['regex_len']}-char regex, {ex['tp_hits']}/{ex['held_out']} held-out "
              f"positives, {ex['fp_hits']}/{ex['negatives']} negatives")
        if hit:
            rep, si, rx = hit
            print(f"  found: rep{rep} seed_index {si} -- all five figures match")
            print(f"  regex: {rx}")
            ok += 5
        else:
            print("  NOT FOUND: no trial satisfies all five figures")
    print(f"\n{ok}/10 Appendix A figures reproduce")
    sys.exit(0 if ok == 10 else 1)


if __name__ == "__main__":
    main()
