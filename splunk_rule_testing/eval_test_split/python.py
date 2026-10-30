import json
import os
from typing import List, Dict, Any, Set, Optional
import agent as harness
from detection import PythonEngine, Colors

from ._splitting import (
    RuleSplit,
    iter_trial_partitions,
    plan_trial_count,
    log_run_header,
    run_result_meta,
)
from ._scoring import score_validated_rule


def setup_tn_index(engine: PythonEngine, files_to_process: List[str]) -> int:
    """Collect, deduplicate and load all TNs into the engine's 'tn' store."""
    TN_STORE = "tn"
    engine.create_store(TN_STORE)

    all_tns: Set[str] = set()
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for tn in data.get("true_negatives", []):
                if tn and tn.strip():
                    all_tns.add(tn)
        except Exception as e:
            print(f"{Colors.WARNING}[WARN] Could not read TNs from {file_path}: {e}{Colors.ENDC}")

    print(f"{Colors.OKBLUE}[INFO] Loading {len(all_tns)} deduplicated True Negatives into engine...{Colors.ENDC}")
    engine.ingest_lines(list(all_tns), TN_STORE)
    return len(all_tns)


def evaluate_file(
    file_path: str,
    api_key: str,
    engine: PythonEngine,
    vertexai: bool = False,
    vertex_project: str = os.environ.get("VERTEX_PROJECT", ""),
    verbose_llm: bool = False,
    global_tn: bool = False,
    no_description: bool = False,
    cross_val: bool = False,
    N: int = 10,
    test_split: Optional[float] = None,
    num_trial: Optional[int] = None,
    master_seed: int = 0,
    rule_split_by_run: Optional[Dict[int, RuleSplit]] = None,
) -> Dict[str, Any]:
    """Evaluate rule generation for a single dataset file using Python mode."""
    print(f"\n{Colors.HEADER}=== Evaluating {os.path.basename(file_path)} (Python Docker) ==={Colors.ENDC}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    true_positives = data.get("true_positives", [])
    true_negatives = data.get("true_negatives", [])
    description = data.get("description", "")

    if not true_positives:
        return {"file": file_path, "status": "Error: No True Positives"}

    num_partitions, _seeds_to_test = plan_trial_count(
        true_positives,
        test_split=test_split,
        num_trial=num_trial,
        cross_val=cross_val,
        N=N,
    )

    TN_STORE = "tn"
    global_tn_set: Set[str] = engine.get_store_content(TN_STORE) if global_tn else set()
    per_seed_results = []

    tp_test: List[str] = []
    tn_test: List[str] = []
    test_count = 0

    for partition in iter_trial_partitions(
        file_path,
        true_positives,
        true_negatives,
        master_seed=master_seed,
        test_split=test_split,
        num_trial=num_trial,
        cross_val=cross_val,
        N=N,
        rule_split_by_run=rule_split_by_run,
    ):
        seed_tp = partition.seed_tp
        held_out_tps = partition.held_out_tps
        tp_test = partition.tp_test
        tn_test = partition.tn_test
        test_count = partition.test_count
        run_idx = partition.run_idx

        log_run_header(partition, held_out_tps)

        GT_STORE = f"gt_eval_{run_idx}"
        engine.create_store(GT_STORE)
        engine.ingest_line(seed_tp, GT_STORE)

        DISCOVERY_STORE = f"discovery_{run_idx}"
        engine.create_store(DISCOVERY_STORE)
        engine.ingest_lines(partition.discovery_logs, DISCOVERY_STORE)

        harness.reset_api_call_count()
        validated_rule, status = harness.generate_rule_for_log(
            seed_tp, GT_STORE, api_key,
            description=description,
            python_engine=engine,
            vertexai=vertexai,
            vertex_project=vertex_project,
            verbose_llm=verbose_llm,
            main_dataset=DISCOVERY_STORE,
            no_description=no_description,
        )
        api_calls = harness.get_api_call_count()

        if not validated_rule:
            per_seed_results.append({
                **run_result_meta(partition),
                "status": f"Generation Failed: {status}",
                "final_rule": None,
                "tpr": None,
                "tp_hits": 0,
                "total_tps": len(held_out_tps),
                "unmatched_tps": list(held_out_tps),
                "fpr": None,
                "fp_hits": 0,
                "total_tns": len(partition.tn_train) if not global_tn else len(global_tn_set),
                "test_tpr": None,
                "test_tp_hits": 0,
                "test_total_tps": len(tp_test),
                "test_unmatched_tps": list(tp_test),
                "test_fpr": None,
                "test_fp_hits": 0,
                "test_total_tns": len(tn_test),
                "api_calls": api_calls,
            })
            continue

        scores = score_validated_rule(
            engine,
            validated_rule,
            partition,
            global_tn=global_tn,
            global_tn_set=global_tn_set,
            test_split=test_split,
        )

        per_seed_results.append({
            **run_result_meta(partition),
            "status": "Success",
            "final_rule": validated_rule,
            "api_calls": api_calls,
            **scores,
        })

    successful = [r for r in per_seed_results if r.get("status") == "Success"]
    measurable = [r for r in successful if r.get("tpr") is not None]
    avg_recall = (sum(r["tpr"] for r in measurable) / len(measurable)) if measurable else None
    total_fp = sum(r.get("fp_hits", 0) for r in successful)
    avg_fp = total_fp / len(successful) if successful else 0
    all_unmatched = []
    for r in successful:
        all_unmatched.extend(r.get("unmatched_tps", []))

    tp_test_recalls = [r.get("test_tpr") for r in successful if r.get("test_tpr") is not None]
    tn_test_fprs = [r.get("test_fpr") for r in successful if r.get("test_fpr") is not None]
    avg_tp_test_recall = (sum(tp_test_recalls) / len(tp_test_recalls)) if tp_test_recalls else None
    avg_tn_test_fpr = (sum(tn_test_fprs) / len(tn_test_fprs)) if tn_test_fprs else None
    total_test_fp = sum(r.get("test_fp_hits", 0) for r in successful)
    all_test_unmatched = []
    for r in successful:
        all_test_unmatched.extend(r.get("test_unmatched_tps", []) if r.get("test_unmatched_tps") else [])
    unique_test_fn = list(set(all_test_unmatched))

    avg_recall_display = f"{avg_recall:.2%}" if avg_recall is not None else "N/A"
    print(f"\n{Colors.OKGREEN}[AGGREGATE] Avg recall: {avg_recall_display} over {len(successful)} successful seed(s), Total FP: {total_fp} (avg {avg_fp:.1f}){Colors.ENDC}")

    if len(per_seed_results) > 1:
        print(f"{Colors.HEADER}{'Run':<8} {'Status':<15} {'Recall':<10} {'FP':<10}{Colors.ENDC}")
        for r in per_seed_results:
            status = r.get("status", "Unknown")
            rec = "N/A" if r.get("status") != "Success" or r.get("tpr") is None else f"{r['tpr']:.2%}"
            fp = str(r.get("fp_hits", 0)) if status == "Success" else "N/A"
            print(f"{r['run_idx']:<8} {status:<15} {rec:<10} {fp:<10}")

    return {
        "file": file_path,
        "status": "Success" if successful else "All runs failed",
        "metrics": {
            "avg_recall": avg_recall,
            "recall": avg_recall,
            "avg_fp_hits": avg_fp,
            "num_partitions": num_partitions if test_split is not None else 1,
            "num_seeds": len(per_seed_results),
            "successful_seeds": len(successful),
            "total_tps": len(true_positives),
            "average_test_true_positive_rate": avg_tp_test_recall,
            "average_test_false_positive_rate": avg_tn_test_fpr,
            "total_test_false_positives": total_test_fp,
            "total_test_false_negatives": len(unique_test_fn),
            "test_holdout": {
                "test_split": test_split,
                "test_count": (test_count if test_split is not None else 0),
                "tp_test_count": len(tp_test),
                "tn_test_count": len(tn_test),
            }
        },
        "trials": [
            {
                **{k: r.get(k) for k in (
                    "partition_idx", "seed_idx", "run_idx", "run_seed", "selected_log_index"
                )},
                "status": r["status"],
                "tpr": r.get("tpr"),
                "tp_hits": r.get("tp_hits"),
                "total_tps": r.get("total_tps"),
                "fpr": r.get("fpr"),
                "fp_hits": r.get("fp_hits"),
                "total_tns": r.get("total_tns"),
                "rule": r.get("final_rule"),
                "test_tpr": r.get("test_tpr"),
                "test_tp_hits": r.get("test_tp_hits"),
                "test_total_tps": r.get("test_total_tps"),
                "test_fpr": r.get("test_fpr"),
                "test_fp_hits": r.get("test_fp_hits"),
                "test_total_tns": r.get("test_total_tns"),
                "api_calls": r.get("api_calls", 0),
            } for r in per_seed_results
        ],
        "per_seed_results": per_seed_results,
        "false_negatives": list(set(all_unmatched)),
    }
