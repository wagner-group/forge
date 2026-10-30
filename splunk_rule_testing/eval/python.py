import json
import os
import random
from typing import List, Dict, Any, Set
import agent as harness
from detection import PythonEngine, Colors
from eval._seeding import derive_seed


# Vertex AI project id. Previously a hardcoded lab project; read from the
# environment so no cloud account id ships in the public artifact. Only used
# when --vertexai is passed.
_DEFAULT_VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")


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
    vertex_project: str = _DEFAULT_VERTEX_PROJECT,
    verbose_llm: bool = False,
    global_tn: bool = False,
    no_description: bool = False,
    cross_val: bool = False,
    N: int = 10,
    master_seed: int = 0,
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

    file_name = os.path.basename(file_path)
    if not cross_val:
        seeds_to_test = true_positives[:1]
    else:
        sample_seed = derive_seed(master_seed, "sample", file_name)
        seeds_to_test = random.Random(sample_seed).sample(
            true_positives, min(len(true_positives), N)
        )

    TN_STORE = "tn"
    global_tn_set: Set[str] = engine.get_store_content(TN_STORE) if global_tn else set()
    per_seed_results = []

    for seed_idx, seed_tp in enumerate(seeds_to_test):
        # held_out_tps = [tp for j, tp in enumerate(true_positives) if j != seed_idx]
        held_out_tps = [tp for tp in true_positives if tp != seed_tp]
        print(f"\n{Colors.OKBLUE}[RUN {seed_idx+1}/{len(seeds_to_test)}] Seed TP {seed_idx} "
              f"({len(held_out_tps)} held-out TPs){Colors.ENDC}")

        GT_STORE = f"gt_eval_{seed_idx}"
        engine.create_store(GT_STORE)
        engine.ingest_line(seed_tp, GT_STORE)

        DISCOVERY_STORE = f"discovery_{seed_idx}"
        engine.create_store(DISCOVERY_STORE)
        # Flat, unlabeled, byte-deduped, deterministically shuffled discovery pool.
        # Cross-file dups remain in the dataset by design; within-file dups were removed
        # at dataset level; collapse any residual TP/TN overlap and remove ingest-order bias.
        all_file_logs = list(dict.fromkeys(true_positives + true_negatives))
        shuffle_seed = derive_seed(master_seed, "shuffle", file_name, seed_idx)
        random.Random(shuffle_seed).shuffle(all_file_logs)
        engine.ingest_lines(all_file_logs, DISCOVERY_STORE)

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
                "seed_index": seed_idx,
                "status": f"Generation Failed: {status}",
                "recall": 0, "tp_hits": 0, "held_out_tps": len(held_out_tps),
                "fp_hits": 0,
                "api_calls": api_calls,
            })
            continue

        if held_out_tps:
            TP_STORE = f"tp_eval_{seed_idx}"
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
            recall = None  # held-out is empty (e.g. single-TP file) — recall is not measurable
            recall_display = "N/A"

        print(f"{Colors.OKGREEN}[RESULT] Recall: {recall_display} ({tp_hits}/{len(held_out_tps)}){Colors.ENDC}")

        matched_tn_raws = engine.get_matched_raws(validated_rule, TN_STORE)
        stripped_matched_raws = {r.strip() for r in matched_tn_raws}

        if global_tn:
            matched_tns = [tn for tn in global_tn_set if tn.strip() in stripped_matched_raws]
        else:
            matched_tns = [tn for tn in true_negatives if tn.strip() in stripped_matched_raws]

        fp_hits = len(matched_tns)
        total_tns = len(global_tn_set) if global_tn else len(true_negatives)
        print(f"{Colors.WARNING if fp_hits > 0 else Colors.OKGREEN}[RESULT] False Positives: {fp_hits}/{total_tns}{Colors.ENDC}")

        per_seed_results.append({
            "seed_index": seed_idx,
            "status": "Success",
            "final_rule": validated_rule,
            "recall": recall,
            "tp_hits": tp_hits,
            "held_out_tps": len(held_out_tps),
            "unmatched_tps": unmatched,
            "fp_hits": fp_hits,
            "total_tns": total_tns,
            "api_calls": api_calls,
        })

    successful = [r for r in per_seed_results if r.get("status") == "Success"]
    measurable = [r for r in successful if r.get("recall") is not None]
    avg_recall = (sum(r["recall"] for r in measurable) / len(measurable)) if measurable else None
    total_tp_hits = sum(r["tp_hits"] for r in successful)
    total_held_out = sum(r["held_out_tps"] for r in successful)
    total_fp = sum(r.get("fp_hits", 0) for r in successful)
    avg_fp = total_fp / len(successful) if successful else 0
    all_unmatched = []
    for r in successful:
        all_unmatched.extend(r.get("unmatched_tps", []))

    avg_recall_display = f"{avg_recall:.2%}" if avg_recall is not None else "N/A"
    print(f"\n{Colors.OKGREEN}[AGGREGATE] Avg recall: {avg_recall_display} over {len(successful)} successful seed(s), Total FP: {total_fp} (avg {avg_fp:.1f}){Colors.ENDC}")

    if len(seeds_to_test) > 1:
        print(f"{Colors.HEADER}{'Trial':<8} {'Status':<15} {'Recall':<10} {'FP':<10}{Colors.ENDC}")
        for r in per_seed_results:
            status = r.get("status", "Unknown")
            rec = "N/A" if r.get("status") != "Success" or r.get("recall") is None else f"{r['recall']:.2%}"
            fp = str(r.get("fp_hits", 0)) if status == "Success" else "N/A"
            print(f"{r['seed_index']:<8} {status:<15} {rec:<10} {fp:<10}")

    return {
        "file": file_path,
        "status": "Success" if successful else "All runs failed",
        "metrics": {
            "avg_recall": avg_recall,
            "recall": avg_recall,
            "avg_fp_hits": avg_fp,
            "num_seeds": len(seeds_to_test),
            "successful_seeds": len(successful),
            "total_tps": len(true_positives),
        },
        "trials": [
            {
                "seed_index": r["seed_index"],
                "status": r["status"],
                "recall": r.get("recall"),
                "tp_hits": r.get("tp_hits"),
                "held_out_tps": r.get("held_out_tps"),
                "fp_hits": r.get("fp_hits"),
                "total_tns": r.get("total_tns"),
                "rule": r.get("final_rule"),
                "api_calls": r.get("api_calls", 0),
            } for r in per_seed_results
        ],
        "per_seed_results": per_seed_results,
        "false_negatives": list(set(all_unmatched)),
    }
