"""Shared RegexEngine scoring for train and test partitions."""
from __future__ import annotations

from typing import Any, Dict, Optional, Set

from detection import RegexEngine, PythonEngine, Colors

Engine = RegexEngine | PythonEngine

from ._splitting import TrialPartition


def score_validated_rule(
    engine: Engine,
    validated_rule: str,
    partition: TrialPartition,
    *,
    global_tn: bool,
    global_tn_set: Set[str],
    test_split: Optional[float],
) -> Dict[str, Any]:
    """Score a regex on held-out train TPs/TNs and optional test holdout."""
    run_idx = partition.run_idx
    held_out_tps = partition.held_out_tps
    tp_test = partition.tp_test
    tn_test = partition.tn_test
    tn_train = partition.tn_train

    TN_STORE = "tn"

    if held_out_tps:
        TP_STORE = f"tp_eval_{run_idx}"
        engine.create_store(TP_STORE)
        engine.ingest_lines(held_out_tps, TP_STORE)
        _, _, tp_hits = engine.test_pattern(validated_rule, TP_STORE)
        matched_tp_raws = engine.get_matched_raws(validated_rule, TP_STORE)
        stripped_tp_raws = {r.strip() for r in matched_tp_raws}
        unmatched = [tp for tp in held_out_tps if tp.strip() not in stripped_tp_raws]
        recall = min(tp_hits / len(held_out_tps), 1.0)
        recall_display = f"{recall:.2%}"
    else:
        tp_hits = 0
        unmatched = []
        recall = None
        recall_display = "N/A"

    print(
        f"{Colors.OKGREEN}[RESULT] Recall: {recall_display} "
        f"({tp_hits}/{len(held_out_tps)}){Colors.ENDC}"
    )

    matched_tn_raws = engine.get_matched_raws(validated_rule, TN_STORE)
    stripped_matched_raws = {r.strip() for r in matched_tn_raws}

    if global_tn:
        matched_tns = [tn for tn in global_tn_set if tn.strip() in stripped_matched_raws]
        total_tns = len(global_tn_set)
    else:
        matched_tns = [tn for tn in tn_train if tn.strip() in stripped_matched_raws]
        total_tns = len(tn_train)
    fp_hits = len(matched_tns)
    fpr = (fp_hits / total_tns) if total_tns > 0 else None
    print(
        f"{Colors.WARNING if fp_hits > 0 else Colors.OKGREEN}"
        f"[RESULT] False Positives: {fp_hits}/{total_tns}{Colors.ENDC}"
    )

    result: Dict[str, Any] = {
        "tpr": recall,
        "tp_hits": tp_hits,
        "total_tps": len(held_out_tps),
        "unmatched_tps": unmatched,
        "fpr": fpr,
        "fp_hits": fp_hits,
        "total_tns": total_tns,
        "test_tpr": None,
        "test_tp_hits": 0,
        "test_total_tps": len(tp_test),
        "test_unmatched_tps": list(tp_test),
        "test_fpr": None,
        "test_fp_hits": 0,
        "test_total_tns": len(tn_test),
    }

    if test_split is not None:
        if tp_test:
            TP_TEST_STORE = f"tp_test_{run_idx}"
            engine.create_store(TP_TEST_STORE)
            engine.ingest_lines(tp_test, TP_TEST_STORE)
            _, _, tp_test_hits = engine.test_pattern(validated_rule, TP_TEST_STORE)
            matched_tp_test_raws = engine.get_matched_raws(validated_rule, TP_TEST_STORE)
            stripped_tp_test_raws = {r.strip() for r in matched_tp_test_raws}
            test_unmatched = [
                tp for tp in tp_test if tp.strip() not in stripped_tp_test_raws
            ]
            tp_test_recall = tp_test_hits / len(tp_test) if len(tp_test) > 0 else None
        else:
            tp_test_hits = 0
            test_unmatched = []
            tp_test_recall = None

        if tn_test:
            TN_TEST_STORE = f"tn_test_{run_idx}"
            engine.create_store(TN_TEST_STORE)
            engine.ingest_lines(tn_test, TN_TEST_STORE)
            matched_tn_test_raws = engine.get_matched_raws(validated_rule, TN_TEST_STORE)
            stripped_matched_tn_test_raws = {r.strip() for r in matched_tn_test_raws}
            matched_tn_test = [
                tn for tn in tn_test if tn.strip() in stripped_matched_tn_test_raws
            ]
            tn_test_fp_hits = len(matched_tn_test)
            tn_test_total = len(tn_test)
            tn_test_fpr = tn_test_fp_hits / tn_test_total if tn_test_total > 0 else None
        else:
            tn_test_fp_hits = 0
            tn_test_total = 0
            tn_test_fpr = None

        result.update(
            {
                "test_tp_hits": tp_test_hits,
                "test_tpr": tp_test_recall,
                "test_unmatched_tps": test_unmatched,
                "test_fp_hits": tn_test_fp_hits,
                "test_total_tns": tn_test_total,
                "test_fpr": tn_test_fpr,
                "test_total_tps": len(tp_test),
            }
        )

    return result
