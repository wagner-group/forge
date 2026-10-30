#!/usr/bin/env python3
"""Generate Claude agent eval directory trees with a global 50/50 train/test split.

Does NOT invoke Claude or score rules — only materializes directories for manual runs.

Split scheme (global 50/50 cut with a protected seed):
  - Per rule: reserve one protected train TP (derive_seed(run_seed, "protected_seed", rule)).
  - Pool all OTHER log lines globally; one 50/50 cut (run_seed = master_seed + run_index).
  - Protected TPs always land on train; remaining lines route per the global cut.
  - Batch seeds shuffled from each rule's full train-side TP set (protected eligible, not forced).

Output layout per (variant, run):
  {output_root}/{variant}/run_{NN}/
    split_dataset/          # labeled ground truth (train/test lists per rule)
    eval/                   # Claude CWD ({{EVAL_ROOT}})
      {pool_dir}/           # 01.json..66.json — bare string arrays
      {batch_dir}/          # batch_001..010/NN.json — seed files
      AGENT_PROMPT_*.md
      {results_dir}/        # empty
      scratch/              # empty
      filename_mapping.md

Three logical directories per run: split_dataset, pool (inside eval), batch (inside eval).
10 runs × 2 variants → 20 run folders (40 including eval/ + split_dataset/ pairs).

Usage:
  python3 generate_claude_eval_run_dirs.py \\
    --dataset-dir ../../dataset/final \\
    --master-seed 161 \\
    --output-root /tmp/claude_eval_gen \\
    --variant both \\
    --pool train \\
    --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

NUM_SLOTS = 66
NUM_BATCHES = 10

Variant = Literal["with_description", "no_description"]
PoolMode = Literal["train", "full"]
Mode = Literal["continuous", "retrospective"]


@dataclass
class LineEntry:
    slot: int
    rule_name: str
    label: Literal["tp", "tn"]
    line: str
    line_index: int


@dataclass
class RuleSlot:
    slot: int
    file_name: str
    file_path: Path
    description: str
    source: dict
    true_positives: List[str]
    true_negatives: List[str]


@dataclass
class RuleSplit:
    slot: RuleSlot
    tp_train: List[str] = field(default_factory=list)
    tp_test: List[str] = field(default_factory=list)
    tn_train: List[str] = field(default_factory=list)
    tn_test: List[str] = field(default_factory=list)


def derive_seed(master_seed: int, domain: str, *identifiers: object) -> int:
    """Same scheme as splunk_rule_testing/eval/_seeding.py (copied to avoid imports)."""
    payload = "|".join([str(master_seed), domain, *(str(i) for i in identifiers)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def run_seed(master_seed: int, run_index: int) -> int:
    """Deterministic seed for one run's global split and batch shuffles."""
    return master_seed + run_index


def slot_label(slot: int) -> str:
    return f"{slot:02d}"


def batch_label(batch_index: int) -> str:
    return f"batch_{batch_index:03d}"


def load_rule_slots(dataset_dir: Path) -> List[RuleSlot]:
    paths = sorted(dataset_dir.glob("*.json"))
    if len(paths) != NUM_SLOTS:
        raise ValueError(
            f"Expected {NUM_SLOTS} JSON files in {dataset_dir}, found {len(paths)}"
        )
    slots: List[RuleSlot] = []
    for i, path in enumerate(paths, start=1):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tps = data.get("true_positives") or []
        tns = data.get("true_negatives") or []
        if not tps:
            raise ValueError(f"No true_positives in {path}")
        slots.append(
            RuleSlot(
                slot=i,
                file_name=path.name,
                file_path=path,
                description=data.get("description", ""),
                source=data.get("source") or {},
                true_positives=list(tps),
                true_negatives=list(tns),
            )
        )
    return slots


def pick_protected_seed_tp(rule: RuleSlot, seed: int) -> str | None:
    """Deterministically reserve one train-side TP before the global cut."""
    if not rule.true_positives:
        return None
    rng = random.Random(derive_seed(seed, "protected_seed", rule.file_name))
    return rule.true_positives[rng.randrange(len(rule.true_positives))]


def global_split(slots: Sequence[RuleSlot], seed: int) -> Tuple[List[RuleSplit], dict]:
    """Global ≈50/50 cut after per-rule protected seed TPs are removed from the pool."""
    protected: Dict[int, str] = {}
    for rule in slots:
        protected_tp = pick_protected_seed_tp(rule, seed)
        if protected_tp is not None:
            protected[rule.slot] = protected_tp

    entries: List[LineEntry] = []
    for rule in slots:
        prot = protected.get(rule.slot)
        # Exclude the protected seed by INDEX, not by value. Excluding by value
        # would drop every duplicate of that line from the cut pool while only
        # one copy is reinserted into tp_train. Dormant today (no file in
        # dataset/final has duplicate TPs) but wrong if the dataset changes.
        prot_idx = rule.true_positives.index(prot) if prot is not None else None
        for idx, line in enumerate(rule.true_positives):
            if idx == prot_idx:
                continue
            entries.append(LineEntry(rule.slot, rule.file_name, "tp", line, idx))
        for idx, line in enumerate(rule.true_negatives):
            entries.append(LineEntry(rule.slot, rule.file_name, "tn", line, idx))

    rng = random.Random(seed)
    rng.shuffle(entries)
    mid = len(entries) // 2
    train_entries = entries[:mid]
    test_entries = entries[mid:]

    by_slot: Dict[int, RuleSplit] = {r.slot: RuleSplit(slot=r) for r in slots}
    for entry in train_entries:
        rs = by_slot[entry.slot]
        if entry.label == "tp":
            rs.tp_train.append(entry.line)
        else:
            rs.tn_train.append(entry.line)
    for entry in test_entries:
        rs = by_slot[entry.slot]
        if entry.label == "tp":
            rs.tp_test.append(entry.line)
        else:
            rs.tn_test.append(entry.line)

    for rule in slots:
        prot = protected.get(rule.slot)
        if prot is not None and prot not in by_slot[rule.slot].tp_train:
            by_slot[rule.slot].tp_train.insert(0, prot)

    splits = [by_slot[r.slot] for r in slots]
    rules_missing_train_tp = [
        s.slot.file_name for s in splits if s.slot.true_positives and not s.tp_train
    ]
    rules_missing_batch_seed = [
        s.slot.file_name
        for s in splits
        if s.slot.true_positives and not compute_batch_seeds(s, seed)
    ]
    manifest = {
        "scheme": "global_50_50_protected_seed",
        "run_seed": seed,
        "protected_seed_count": len(protected),
        "total_lines": len(entries) + len(protected),
        "pool_lines_for_cut": len(entries),
        "train_lines": len(train_entries) + len(protected),
        "test_lines": len(test_entries),
        "rules_no_test_tp": [
            s.slot.file_name for s in splits if s.tp_train and not s.tp_test
        ],
        "rules_no_test_tn": [
            s.slot.file_name for s in splits if s.tn_train and not s.tn_test
        ],
        "rules_no_train_tp": [s.slot.file_name for s in splits if not s.tp_train],
        "rules_no_train_tn": [s.slot.file_name for s in splits if not s.tn_train],
        "rules_missing_train_tp_with_tps": rules_missing_train_tp,
        "rules_missing_batch_seed_with_tps": rules_missing_batch_seed,
    }
    return splits, manifest


def no_split(slots: Sequence[RuleSlot], seed: int) -> Tuple[List[RuleSplit], dict]:
    """Retrospective: no train/test cut.

    Train/test only has meaning under the continuous paradigm. Retrospective
    scores a rule against everything except its own seed, so every line stays
    available and batch seeds are drawn from each rule's full TP list. That
    reproduces the batch shape of claude_eval_corrected (66, 56, 33, 26, ...)
    rather than the split-depleted (66, 42, 22, 14, ...).
    """
    splits = [
        RuleSplit(
            slot=rule,
            tp_train=list(rule.true_positives),
            tn_train=list(rule.true_negatives),
        )
        for rule in slots
    ]
    total_tp = sum(len(r.true_positives) for r in slots)
    total_tn = sum(len(r.true_negatives) for r in slots)
    manifest = {
        "scheme": "retrospective_no_split",
        "run_seed": seed,
        "total_lines": total_tp + total_tn,
        "total_tps": total_tp,
        "total_tns": total_tn,
        "rules_missing_batch_seed_with_tps": [
            s.slot.file_name
            for s in splits
            if s.slot.true_positives and not compute_batch_seeds(s, seed)
        ],
    }
    return splits, manifest


def compute_batch_seeds(rule: RuleSplit, seed: int) -> List[str]:
    """Up to 10 train-side TPs, shuffled deterministically per rule."""
    train_tps = list(rule.tp_train)
    if not train_tps:
        return []
    rng = random.Random(derive_seed(seed, "batch_seeds", rule.slot.file_name))
    rng.shuffle(train_tps)
    return train_tps[:NUM_BATCHES]


def pool_lines(rule: RuleSplit, pool_mode: PoolMode) -> List[str]:
    if pool_mode == "full":
        return list(dict.fromkeys(rule.slot.true_positives + rule.slot.true_negatives))
    return list(dict.fromkeys(rule.tp_train + rule.tn_train))


def write_json(path: Path, payload: object, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_filename_mapping(slots: Sequence[RuleSlot]) -> str:
    lines = [
        "# Filename Mapping",
        "",
        "Maps numbered filenames to source rule names (alphabetical by filename).",
        "",
        "| File | Rule |",
        "| --- | --- |",
    ]
    for rule in slots:
        lines.append(f"| `{slot_label(rule.slot)}.json` | `{rule.file_name[:-5]}` |")
    lines.append("")
    return "\n".join(lines)


# Legacy absolute prompt roots from the machine the published runs were
# generated on, rewritten to {{EVAL_ROOT}} when an as-run template is replayed.
# These are the DEFAULTS, so behaviour with no environment set is identical to
# the published runs and the `leftovers` guard below still fires. Override with
# a JSON array in $LEGACY_PROMPT_PATH_PATTERNS / $LEGACY_PROMPT_PATH_LITERALS
# (a ":"-separated list would corrupt the regexes, which contain "(?:").
# The as-run prompt templates under results/claude/ carry the absolute working
# directory of the machine the published runs were generated on. These patterns
# rewrite any such prefix to {{EVAL_ROOT}}. The account name is matched as a
# character class rather than written out, so no personal path is recorded here;
# the patterns are correspondingly broader than the original run roots.
DEFAULT_LEGACY_PROMPT_PATH_PATTERNS = (
    r"/home/[^/\s]+/(?:realtime_)?agent_run_with_desc[\w./-]*",
    r"/home/[^/\s]+/log_eval_claude_no_desc_loop[\w]*/eval_\d+",
    r"/data2/[^/\s]+/claude_eval_\w+/(?:no|with)_description/run_\d+/eval",
)
# A remaining absolute home directory or data mount means a rewrite was missed.
# Broader than the patterns above on purpose: this is the guard, not the rewrite.
DEFAULT_LEGACY_PROMPT_PATH_LITERALS = ("/home/", "/data2/")


def _legacy_list(env_name, default):
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    return tuple(json.loads(raw))


LEGACY_PROMPT_PATH_PATTERNS = _legacy_list(
    "LEGACY_PROMPT_PATH_PATTERNS", DEFAULT_LEGACY_PROMPT_PATH_PATTERNS)
LEGACY_PROMPT_PATH_LITERALS = _legacy_list(
    "LEGACY_PROMPT_PATH_LITERALS", DEFAULT_LEGACY_PROMPT_PATH_LITERALS)


def resolve_prompt(
    template_path: Path,
    eval_root: Path,
    variant: Variant,
    run_index: int,
) -> str:
    text = template_path.read_text(encoding="utf-8")
    eval_str = str(eval_root.resolve())

    # Legacy absolute paths FIRST, then the marker.
    #
    # Order matters: substituting {{EVAL_ROOT}} first means these patterns run
    # over their own output, so an eval root whose own path contains a legacy
    # prefix gets rewritten a second time. Regenerating into the archive's own
    # location produced a doubled, non-existent path with no validation.
    # Applied in order. The third pattern exists because per-run copies
    # recovered from the archive are already instantiated and carry their own
    # absolute eval root instead of the {{EVAL_ROOT}} marker.
    for pattern in LEGACY_PROMPT_PATH_PATTERNS:
        text = re.sub(pattern, "{{EVAL_ROOT}}", text)
    # Check for unrewritten legacy paths BEFORE substituting the marker. Checking
    # afterwards cannot work: a legitimate eval root may itself contain one of
    # these prefixes (regenerating into the archive's own location does), so the
    # check would fire on correctly substituted output -- the same confusion
    # between "a path we wrote" and "a path we failed to rewrite" that this
    # function's ordering bug was about.
    leftovers = [pat for pat in LEGACY_PROMPT_PATH_LITERALS if pat in text]
    if leftovers:
        raise ValueError(
            f"resolve_prompt left unrewritten legacy paths {leftovers} in "
            f"{template_path}; the emitted prompt would point somewhere wrong")

    text = text.replace("{{EVAL_ROOT}}", eval_str)

    if "{{EVAL_ROOT}}" in text:
        raise ValueError(f"{template_path}: unresolved {{{{EVAL_ROOT}}}} marker remains")
    if eval_str not in text:
        raise ValueError(
            f"{template_path}: emitted prompt never mentions the eval root {eval_str}")
    if variant == "no_description":
        rep_name = f"rep_{run_index:02d}_no_description"
        text = text.replace("rep_01_no_description", rep_name)
    return text


def variant_layout(variant: Variant, run_index: int) -> dict:
    if variant == "with_description":
        return {
            "pool_dir": "randomize_pre_filter",
            "batch_dir": "single_positive_log_with_description",
            "results_dir": "codex_results_loop",
            "prompt_name": "AGENT_PROMPT_LOOP.claude.md",
        }
    return {
        "pool_dir": "randomized",
        "batch_dir": f"rep_{run_index:02d}_no_description",
        "results_dir": "codex_results_no_description",
        "prompt_name": "AGENT_PROMPT_NO_DESCRIPTION_LOOP.claude.md",
    }


def seed_payload(variant: Variant, description: str, true_positive: str) -> dict:
    if variant == "with_description":
        return {"description": description, "true_positive": true_positive}
    return {"true_positive": true_positive}


def write_split_dataset(
    out_dir: Path,
    splits: Sequence[RuleSplit],
    manifest: dict,
    run_index: int,
    dry_run: bool,
) -> None:
    manifest_out = {
        **manifest,
        "run_index": run_index,
    }
    write_json(out_dir / "split_manifest.json", manifest_out, dry_run)
    for rule in splits:
        payload = {
            "description": rule.slot.description,
            "source": rule.slot.source,
            "true_positives_train": rule.tp_train,
            "true_positives_test": rule.tp_test,
            "true_negatives_train": rule.tn_train,
            "true_negatives_test": rule.tn_test,
            "_split_meta": {
                "run_index": run_index,
                "run_seed": manifest["run_seed"],
                # Read from the manifest, not hardcoded: hardcoding silently
                # mislabels every file the moment this is called for a mode
                # whose scheme differs.
                "scheme": manifest.get("scheme", "global_50_50_protected_seed"),
            },
        }
        write_json(out_dir / rule.slot.file_name, payload, dry_run)


def write_pool_dir(
    out_dir: Path,
    splits: Sequence[RuleSplit],
    pool_mode: PoolMode,
    dry_run: bool,
) -> None:
    for rule in splits:
        lines = pool_lines(rule, pool_mode)
        write_json(out_dir / f"{slot_label(rule.slot.slot)}.json", lines, dry_run)


def write_batch_dir(
    out_dir: Path,
    splits: Sequence[RuleSplit],
    variant: Variant,
    seed: int,
    dry_run: bool,
) -> dict:
    batch_counts: Dict[str, int] = {}
    for batch_idx in range(1, NUM_BATCHES + 1):
        batch_counts[batch_label(batch_idx)] = 0

    for rule in splits:
        seeds = compute_batch_seeds(rule, seed)
        for batch_idx, tp in enumerate(seeds, start=1):
            batch_counts[batch_label(batch_idx)] += 1
            payload = seed_payload(variant, rule.slot.description, tp)
            write_json(
                out_dir / batch_label(batch_idx) / f"{slot_label(rule.slot.slot)}.json",
                payload,
                dry_run,
            )
    return batch_counts


def generate_run(
    *,
    slots: Sequence[RuleSlot],
    variant: Variant,
    run_index: int,
    master_seed: int,
    pool_mode: PoolMode,
    mode: Mode,
    run_root: Path,
    prompt_template: Path,
    dry_run: bool,
) -> dict:
    seed = run_seed(master_seed, run_index)
    if mode == "retrospective":
        splits, manifest = no_split(slots, seed)
    else:
        splits, manifest = global_split(slots, seed)
    layout = variant_layout(variant, run_index)

    split_dir = run_root / "split_dataset"
    eval_dir = run_root / "eval"
    pool_dir = eval_dir / layout["pool_dir"]
    batch_dir = eval_dir / layout["batch_dir"]
    results_dir = eval_dir / layout["results_dir"]
    scratch_dir = eval_dir / "scratch"

    # split_dataset/ holds train/test ground truth, which is meaningless when
    # there is no split; retrospective scores against dataset/final directly.
    if mode == "continuous":
        write_split_dataset(split_dir, splits, manifest, run_index, dry_run)
    write_pool_dir(pool_dir, splits, pool_mode, dry_run)
    batch_counts = write_batch_dir(batch_dir, splits, variant, seed, dry_run)

    prompt_text = resolve_prompt(prompt_template, eval_dir, variant, run_index)
    write_text(eval_dir / layout["prompt_name"], prompt_text, dry_run)
    write_text(eval_dir / "filename_mapping.md", build_filename_mapping(slots), dry_run)

    if not dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

    return {
        "variant": variant,
        "run_index": run_index,
        "run_root": str(run_root),
        "eval_root": str(eval_dir),
        "run_seed": seed,
        "pool_mode": pool_mode,
        "mode": mode,
        "manifest": manifest,
        "batch_counts": batch_counts,
        "layout": layout,
    }


TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"


def default_prompt_template(variant: Variant) -> Path:
    """Masters derived from the archived per-run copies.

    NOT byte-identical to the archive. Two deliberate deltas:
      * the absolute eval root is replaced by the {{EVAL_ROOT}} marker, and
      * four lines rename Claude -> Codex, including the results directory
        (`claude_results_*` -> `codex_results_*`) and, inside the refinement-loop
        step, "you (Claude)" -> "you (Codex)".

    The rename was requested deliberately, but it is a prompt delta against the
    published Claude runs and must be disclosed in any comparison that treats
    the two as having received the same prompt.
    """
    if variant == "with_description":
        return TEMPLATE_DIR / "AGENT_PROMPT_LOOP.claude.md"
    return TEMPLATE_DIR / "AGENT_PROMPT_NO_DESCRIPTION_LOOP.claude.md"


def parse_variants(raw: str) -> List[Variant]:
    if raw == "both":
        return ["with_description", "no_description"]
    if raw in ("with_description", "no_description"):
        return [raw]  # type: ignore[list-item]
    raise argparse.ArgumentTypeError(
        f"variant must be with_description, no_description, or both (got {raw!r})"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate Claude eval directories with global 50/50 train/test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=repo_root / "dataset" / "final",
        help="Master labeled dataset (66 JSON files)",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        required=True,
        help="Base seed; run i uses master_seed + i",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs per variant (batch_001..batch_010 seeds)",
    )
    parser.add_argument(
        "--variant",
        type=parse_variants,
        default="both",
        help="with_description, no_description, or both",
    )
    parser.add_argument(
        "--mode",
        choices=("continuous", "retrospective"),
        default="continuous",
        help=(
            "continuous: global 50/50 train/test split (writes split_dataset/). "
            "retrospective: no split, seeds drawn from all TPs, no split_dataset/"
        ),
    )
    parser.add_argument(
        "--pool",
        choices=("train", "full"),
        default=None,
        help=(
            "Pool content: train-half only or full master TP∪TN per rule. "
            "Default: train for continuous, full for retrospective"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root output directory",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=None,
        help="Override prompt markdown (default: frozen template per variant)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs and split stats without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing run directories before writing",
    )
    args = parser.parse_args()

    if args.pool is None:
        args.pool = "full" if args.mode == "retrospective" else "train"
    if args.mode == "retrospective" and args.pool == "train":
        # tp_train/tn_train are the whole rule under no_split, so this is the
        # same set; keep it explicit rather than silently equivalent.
        args.pool = "full"

    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    slots = load_rule_slots(dataset_dir)

    summaries: List[dict] = []
    for variant in args.variant:
        prompt_path = args.prompt_template or default_prompt_template(variant)
        if not prompt_path.is_file():
            raise SystemExit(f"Prompt template not found: {prompt_path}")

        for run_index in range(1, args.num_runs + 1):
            run_root = output_root / variant / f"run_{run_index:02d}"
            if run_root.exists():
                if args.force and not args.dry_run:
                    shutil.rmtree(run_root)
                elif not args.dry_run:
                    raise SystemExit(
                        f"Refusing to overwrite {run_root} (pass --force)"
                    )

            summary = generate_run(
                slots=slots,
                variant=variant,
                run_index=run_index,
                master_seed=args.master_seed,
                pool_mode=args.pool,
                mode=args.mode,
                run_root=run_root,
                prompt_template=prompt_path,
                dry_run=args.dry_run,
            )
            summaries.append(summary)

            if args.dry_run:
                m = summary["manifest"]
                bc = summary["batch_counts"]
                print(f"\n=== {variant} run_{run_index:02d} (seed={summary['run_seed']}) ===")
                print(f"  eval_root: {summary['eval_root']}")
                print(f"  mode: {args.mode}")
                if args.mode == "continuous":
                    print(f"  split_dataset: {run_root / 'split_dataset'}")
                print(f"  pool: {summary['layout']['pool_dir']}/ ({args.pool} mode)")
                print(f"  batch: {summary['layout']['batch_dir']}/")
                if args.mode == "retrospective":
                    print(
                        f"  lines: tps={m['total_tps']} tns={m['total_tns']} "
                        f"(total={m['total_lines']}, no split)"
                    )
                else:
                    print(
                        f"  global lines: train={m['train_lines']} test={m['test_lines']} "
                        f"(total={m['total_lines']}, cut_pool={m['pool_lines_for_cut']}, "
                        f"protected={m['protected_seed_count']})"
                    )
                    print(f"  rules_no_test_tp: {len(m['rules_no_test_tp'])}")
                    print(f"  rules_no_test_tn: {len(m['rules_no_test_tn'])}")
                    print(f"  rules_no_train_tp: {len(m['rules_no_train_tp'])}")
                    print(
                        f"  rules_missing_train_tp (has TPs): "
                        f"{len(m['rules_missing_train_tp_with_tps'])}"
                    )
                    if m["rules_missing_train_tp_with_tps"]:
                        print(f"    WARN train: {m['rules_missing_train_tp_with_tps'][:5]}")
                print(
                    f"  rules_missing_batch_seed (has TPs): "
                    f"{len(m['rules_missing_batch_seed_with_tps'])}"
                )
                if m["rules_missing_batch_seed_with_tps"]:
                    print(f"    WARN batch: {m['rules_missing_batch_seed_with_tps'][:5]}")
                print(f"  batch sizes: {bc}")

    if args.dry_run:
        print(
            f"\n[dry-run] Would create {len(summaries)} run folders under {output_root}"
        )
    else:
        print(f"Created {len(summaries)} run folders under {output_root}")


if __name__ == "__main__":
    main()
