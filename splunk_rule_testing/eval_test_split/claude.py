import json
import os
from typing import List, Dict, Any, Set, Optional

from detection import RegexEngine, Colors
from claude.prompts import (
    load_per_agent_prompt_template,
    mapping_path,
    prompt_md_path,
    resolve_numeric_id,
    rule_name_from_file_path,
)
from claude.workspace import prepare_trial_workspace
from claude.agent import run_claude_code_trial
from claude.recovery import is_max_turns_failure, recover_last_matching_regex

from .regex import setup_tn_index
from ._splitting import (
    RuleSplit,
    iter_trial_partitions,
    plan_trial_count,
    log_run_header,
    run_result_meta,
)
from ._scoring import score_validated_rule


def evaluate_file(
    file_path: str,
    api_key: str,
    engine: RegexEngine,
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
    claude_bin: str = "claude",
    claude_model: str = "claude-sonnet-4-6",
    claude_dataset_root: str = "",
    workspaces_dir: str = "",
    claude_proxy_url: Optional[str] = None,
    claude_proxy_key: Optional[str] = None,
    claude_use_gemini_proxy: bool = True,
) -> Dict[str, Any]:
    """Evaluate rule generation for a single dataset file using Claude Code."""
    print(f"\n{Colors.HEADER}=== Evaluating {os.path.basename(file_path)} (claude) ==={Colors.ENDC}")

    dataset_root = claude_dataset_root or os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "log_rules_dataset_final")
    )
    map_path = mapping_path(dataset_root)
    prompt_path = prompt_md_path(dataset_root)
    prompt_template = load_per_agent_prompt_template(prompt_path)

    rule_name = rule_name_from_file_path(file_path)
    numeric_id = resolve_numeric_id(rule_name, map_path)
    if not numeric_id:
        return {
            "file": file_path,
            "status": f"Error: No numeric mapping for rule '{rule_name}'",
        }

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    true_positives = data.get("true_positives", [])
    true_negatives = data.get("true_negatives", [])

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

    base_name = os.path.splitext(os.path.basename(file_path))[0]

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
        held_out_tps = partition.held_out_tps
        tp_test = partition.tp_test
        tn_test = partition.tn_test
        test_count = partition.test_count
        batch_id = f"t{partition.run_idx:03d}"

        log_run_header(partition, held_out_tps)

        trial_workspace = os.path.join(
            workspaces_dir,
            base_name,
            f"trial_{partition.partition_idx + 1:03d}_seed_{partition.seed_idx:03d}",
        )
        prepare_trial_workspace(
            trial_workspace,
            batch_id,
            numeric_id.zfill(2),
            partition.seed_tp,
            partition.discovery_logs,
        )

        claude_log = os.path.join(trial_workspace, "claude_run.log")
        entry, status = run_claude_code_trial(
            trial_workspace,
            batch_id,
            numeric_id.zfill(2),
            prompt_template=prompt_template,
            claude_bin=claude_bin,
            claude_model=claude_model,
            verbose=verbose_llm,
            log_path=claude_log,
            proxy_url=claude_proxy_url,
            proxy_key=claude_proxy_key,
            use_gemini_proxy=claude_use_gemini_proxy,
        )

        refinement_iterations = (entry or {}).get("refinement_iterations")
        api_calls = 0  # Claude Code does not expose API call count via CLI

        validated_rule = None
        max_turns_recovery = False
        recovery_source = None

        if status == "Success":
            validated_rule = entry["regex"]
        elif is_max_turns_failure(status, claude_log):
            recovered, recovery_source = recover_last_matching_regex(
                trial_workspace,
                batch_id,
                partition.seed_tp,
                claude_log,
            )
            if recovered:
                validated_rule = recovered
                max_turns_recovery = True
                print(
                    f"{Colors.WARNING}[RECOVERY] max_turns: using last seed-matching "
                    f"regex from {recovery_source}{Colors.ENDC}"
                )

        if validated_rule is None:
            per_seed_results.append({
                **run_result_meta(partition),
                "status": status,
                "final_rule": (entry or {}).get("regex"),
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
                "refinement_iterations": refinement_iterations,
                "max_turns_recovery": False,
                "recovery_source": None,
                "workspace": trial_workspace,
            })
            continue
        print(
            f"{Colors.OKGREEN}[RULE] {validated_rule[:120]}"
            f"{'...' if len(validated_rule) > 120 else ''}{Colors.ENDC}"
        )

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
            "refinement_iterations": refinement_iterations,
            "max_turns_recovery": max_turns_recovery,
            "recovery_source": recovery_source,
            "workspace": trial_workspace,
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
    print(
        f"\n{Colors.OKGREEN}[AGGREGATE] Avg recall: {avg_recall_display} "
        f"over {len(successful)} successful seed(s), Total FP: {total_fp} (avg {avg_fp:.1f}){Colors.ENDC}"
    )

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
            },
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
                "refinement_iterations": r.get("refinement_iterations"),
                "max_turns_recovery": r.get("max_turns_recovery", False),
                "recovery_source": r.get("recovery_source"),
            }
            for r in per_seed_results
        ],
        "per_seed_results": per_seed_results,
        "false_negatives": list(set(all_unmatched)),
    }
