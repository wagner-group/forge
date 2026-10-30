"""Shared test-split partition logic (global protected-seed scheme).

Matches scripts/eval_analysis/generate_claude_eval_run_dirs.py:
  - Per rule: protected train TP via derive_seed(run_seed, "protected_seed", file_name)
  - Pool remaining lines; shuffle with run_seed; hold out test_split fraction
  - Batch seeds: shuffle tp_train via derive_seed(run_seed, "batch_seeds", file_name)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

from ._seeding import derive_seed

GLOBAL_DATASET_SIZE = 66


@dataclass
class RuleSlot:
    slot: int
    file_name: str
    file_path: str
    description: str
    true_positives: List[str]
    true_negatives: List[str]


@dataclass
class RuleSplit:
    slot: RuleSlot
    tp_train: List[str] = field(default_factory=list)
    tp_test: List[str] = field(default_factory=list)
    tn_train: List[str] = field(default_factory=list)
    tn_test: List[str] = field(default_factory=list)


@dataclass
class LineEntry:
    slot: int
    file_name: str
    label: Literal["tp", "tn"]
    line: str
    line_index: int


@dataclass
class TrialPartition:
    partition_idx: int
    seed_idx: int
    run_idx: int
    run_seed: int
    num_partitions: int
    num_seeds_in_partition: int
    selected_log_index: int
    seed_tp: str
    tp_train: List[str]
    tn_train: List[str]
    tp_test: List[str]
    tn_test: List[str]
    held_out_tps: List[str]
    discovery_logs: List[str]
    test_count: int = 0


def run_seed(master_seed: int, run_index: int) -> int:
    """Deterministic seed for one partition run (1-based run_index)."""
    return master_seed + run_index


def load_rule_slots(file_paths: Sequence[str]) -> List[RuleSlot]:
    """Load dataset JSON files as RuleSlot records (sorted by path)."""
    slots: List[RuleSlot] = []
    for i, path in enumerate(sorted(file_paths), start=1):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tps = data.get("true_positives") or []
        tns = data.get("true_negatives") or []
        if not tps:
            raise ValueError(f"No true_positives in {path}")
        slots.append(
            RuleSlot(
                slot=i,
                file_name=Path(path).name,
                file_path=path,
                description=data.get("description", ""),
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


def global_split(
    slots: Sequence[RuleSlot],
    seed: int,
    test_split: float,
) -> Tuple[List[RuleSplit], dict]:
    """Global train/test cut after per-rule protected seed TPs."""
    protected: Dict[int, str] = {}
    for rule in slots:
        protected_tp = pick_protected_seed_tp(rule, seed)
        if protected_tp is not None:
            protected[rule.slot] = protected_tp

    entries: List[LineEntry] = []
    for rule in slots:
        prot = protected.get(rule.slot)
        for idx, line in enumerate(rule.true_positives):
            if line == prot:
                continue
            entries.append(LineEntry(rule.slot, rule.file_name, "tp", line, idx))
        for idx, line in enumerate(rule.true_negatives):
            entries.append(LineEntry(rule.slot, rule.file_name, "tn", line, idx))

    rng = random.Random(seed)
    rng.shuffle(entries)
    if entries:
        test_count = max(1, min(int(round(len(entries) * test_split)), len(entries) - 1))
    else:
        test_count = 0
    train_entries = entries[:-test_count] if test_count else entries
    test_entries = entries[-test_count:] if test_count else []

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
    manifest = {
        "scheme": "global_protected_seed",
        "run_seed": seed,
        "test_split": test_split,
        "protected_seed_count": len(protected),
        "total_lines": len(entries) + len(protected),
        "pool_lines_for_cut": len(entries),
        "train_lines": len(train_entries) + len(protected),
        "test_lines": len(test_entries),
        "test_count": test_count,
    }
    return splits, manifest


def build_splits_for_run(
    slots: Sequence[RuleSlot],
    run_seed_val: int,
    test_split: float,
) -> Tuple[Dict[str, RuleSplit], dict]:
    """Return {file_name: RuleSplit} and manifest for one partition run."""
    splits, manifest = global_split(slots, run_seed_val, test_split)
    by_name = {s.slot.file_name: s for s in splits}
    return by_name, manifest


def build_splits_by_run(
    slots: Sequence[RuleSlot],
    *,
    master_seed: int,
    num_trial: int,
    test_split: float,
) -> Dict[int, Dict[str, RuleSplit]]:
    """Precompute splits for run_index 1..num_trial."""
    out: Dict[int, Dict[str, RuleSplit]] = {}
    for run_index in range(1, num_trial + 1):
        rs = run_seed(master_seed, run_index)
        by_name, _ = build_splits_for_run(slots, rs, test_split)
        out[run_index] = by_name
    return out


def build_per_file_splits_by_run(
    rule: RuleSlot,
    *,
    master_seed: int,
    num_trial: int,
    test_split: float,
) -> Dict[int, Dict[str, RuleSplit]]:
    """Per-file fallback: same algorithm with a single-rule slot list."""
    return build_splits_by_run(
        [rule],
        master_seed=master_seed,
        num_trial=num_trial,
        test_split=test_split,
    )


def use_global_split_mode(*, test_split: Optional[float], dir_file_count: int, is_dir: bool) -> bool:
    return test_split is not None and is_dir and dir_file_count == GLOBAL_DATASET_SIZE


def compute_batch_seeds(rule: RuleSplit, seed: int, n_cap: int) -> List[str]:
    """Up to n_cap train-side TPs, shuffled deterministically per rule."""
    train_tps = list(rule.tp_train)
    if not train_tps:
        return []
    rng = random.Random(derive_seed(seed, "batch_seeds", rule.slot.file_name))
    rng.shuffle(train_tps)
    return train_tps[:n_cap]


def rule_splits_for_file(
    splits_by_run: Dict[int, Dict[str, RuleSplit]],
    file_name: str,
) -> Dict[int, RuleSplit]:
    """Extract per-run RuleSplit for one file from precomputed splits."""
    return {
        run_index: by_name[file_name]
        for run_index, by_name in splits_by_run.items()
        if file_name in by_name
    }


def run_result_meta(partition: TrialPartition) -> dict:
    return {
        "partition_idx": partition.partition_idx,
        "seed_idx": partition.seed_idx,
        "run_idx": partition.run_idx,
        "run_seed": partition.run_seed,
        "selected_log_index": partition.selected_log_index,
    }


def log_run_header(partition: TrialPartition, held_out_tps: List[str]) -> None:
    from detection import Colors

    print(
        f"\n{Colors.OKBLUE}[RUN {partition.partition_idx + 1}/{partition.num_partitions}] "
        f"[SEED {partition.seed_idx + 1}/{partition.num_seeds_in_partition}] "
        f"run_seed={partition.run_seed} "
        f"Input TP index {partition.selected_log_index} "
        f"({len(held_out_tps)} train-heldout TPs){Colors.ENDC}"
    )


def plan_trial_count(
    true_positives: List[str],
    *,
    test_split: Optional[float],
    num_trial: Optional[int],
    cross_val: bool,
    N: int,
) -> Tuple[int, Union[List[Optional[None]], List[str]]]:
    """Return (num_partitions, seeds_to_test) for display / cross-val."""
    if test_split is not None:
        num = num_trial if num_trial is not None else 1
        num = max(1, num)
        return num, [None] * num
    if not cross_val:
        seeds = true_positives[:1]
    else:
        seeds = random.Random(42).sample(true_positives, min(len(true_positives), N))
    return len(seeds), seeds


def iter_trial_partitions(
    file_path: str,
    true_positives: List[str],
    true_negatives: List[str],
    *,
    master_seed: int = 0,
    test_split: Optional[float] = None,
    num_trial: Optional[int] = None,
    cross_val: bool = False,
    N: int = 10,
    rule_split_by_run: Optional[Dict[int, RuleSplit]] = None,
) -> Iterator[TrialPartition]:
    """Yield one TrialPartition per agent run (partition x batch seed)."""
    file_name = Path(file_path).name
    num_partitions, seeds_to_test = plan_trial_count(
        true_positives,
        test_split=test_split,
        num_trial=num_trial,
        cross_val=cross_val,
        N=N,
    )

    run_idx = 0

    if test_split is not None and rule_split_by_run is not None:
        for run_index in range(1, num_partitions + 1):
            rule_split = rule_split_by_run.get(run_index)
            if rule_split is None:
                continue
            rs = run_seed_val = run_seed(master_seed, run_index)
            batch_seeds = compute_batch_seeds(rule_split, rs, N)
            num_seeds_in_partition = len(batch_seeds)
            all_file_logs = list(dict.fromkeys(rule_split.tp_train + rule_split.tn_train))
            random.Random(42).shuffle(all_file_logs)
            test_count = len(rule_split.tp_test) + len(rule_split.tn_test)

            for seed_idx, seed_tp in enumerate(batch_seeds):
                selected_log_index = (
                    true_positives.index(seed_tp) if seed_tp in true_positives else -1
                )
                held_out_tps = [tp for tp in rule_split.tp_train if tp != seed_tp]
                yield TrialPartition(
                    partition_idx=run_index - 1,
                    seed_idx=seed_idx,
                    run_idx=run_idx,
                    run_seed=run_seed_val,
                    num_partitions=num_partitions,
                    num_seeds_in_partition=num_seeds_in_partition,
                    selected_log_index=selected_log_index,
                    seed_tp=seed_tp,
                    tp_train=list(rule_split.tp_train),
                    tn_train=list(rule_split.tn_train),
                    tp_test=list(rule_split.tp_test),
                    tn_test=list(rule_split.tn_test),
                    held_out_tps=held_out_tps,
                    discovery_logs=all_file_logs,
                    test_count=test_count,
                )
                run_idx += 1
        return

    for seed_idx, seed_tp in enumerate(seeds_to_test):
        selected_log_index = true_positives.index(seed_tp)
        tp_train = [tp for tp in true_positives if tp != seed_tp]
        tn_train = list(true_negatives)
        held_out_tps = list(tp_train)
        all_file_logs = list(dict.fromkeys(tp_train + tn_train))
        random.Random(42).shuffle(all_file_logs)

        yield TrialPartition(
            partition_idx=0,
            seed_idx=seed_idx,
            run_idx=run_idx,
            run_seed=master_seed,
            num_partitions=1,
            num_seeds_in_partition=len(seeds_to_test),
            selected_log_index=selected_log_index,
            seed_tp=seed_tp,
            tp_train=tp_train,
            tn_train=tn_train,
            tp_test=[],
            tn_test=[],
            held_out_tps=held_out_tps,
            discovery_logs=all_file_logs,
            test_count=0,
        )
        run_idx += 1
