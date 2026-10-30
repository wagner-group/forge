import json
import os
import time
import uuid
import random
from typing import List, Dict, Any, Set, Optional
import agent as harness
from detection import SplunkEngine, Colors

from ._splitting import (
    RuleSplit,
    iter_trial_partitions,
    plan_trial_count,
    log_run_header,
    run_result_meta,
)


def setup_evaluation_index(index_name: str, splunk_engine: SplunkEngine):
    print(f"{Colors.OKBLUE}[INFO] Setting up evaluation index '{index_name}'...{Colors.ENDC}")
    splunk_engine.delete_store(index_name)
    time.sleep(2)
    splunk_engine.create_store(index_name)
    time.sleep(2)


def setup_tn_index(
    tn_index: str,
    files_to_process: List[str],
    splunk_engine: SplunkEngine,
    sourcetype: Optional[str],
    api_key: str,
) -> int:
    print(f"\n{Colors.HEADER}[INFO] Setting up global TN index '{tn_index}'...{Colors.ENDC}")
    setup_evaluation_index(tn_index, splunk_engine)

    all_tns: Set[str] = set()
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for tn in data.get("true_negatives", []):
                all_tns.add(tn)
        except Exception as e:
            print(f"{Colors.WARNING}[WARN] Could not read TNs from {file_path}: {e}{Colors.ENDC}")

    all_tns -= {"", None}
    all_tns = {tn for tn in all_tns if tn and tn.strip()}

    print(f"{Colors.OKBLUE}[INFO] Ingesting {len(all_tns)} deduplicated True Negatives in batches...{Colors.ENDC}")
    st = sourcetype or "generic"
    tn_list = list(all_tns)
    splunk_engine.ingest_lines(tn_list, tn_index, st, "eval_host", batch_size=100)

    print(f"{Colors.OKBLUE}[INFO] Waiting for TN ingestion to settle...{Colors.ENDC}")
    time.sleep(5)
    return len(all_tns)


def _swap_index(spl: str, old_index: str, new_index: str) -> str:
    import re
    result = re.sub(
        rf'index\s*=\s*["\']?{re.escape(old_index)}["\']?',
        f'index={new_index}',
        spl
    )
    if f"index={new_index}" not in result:
        if "search" in result:
            result = result.replace("search", f"search index={new_index} ", 1)
        else:
            result = f"search index={new_index} {result}"
    return result


def _strip_aggregation(spl: str) -> str:
    import re
    stripped = re.split(r'\|\s*(stats|table|sort|head|tail|dedup|top|rare)\b', spl, maxsplit=1)[0]
    return stripped.strip()


def _count_raw_events(base_spl: str, splunk_engine: SplunkEngine) -> int:
    count_search = _strip_aggregation(base_spl) + " | stats count"
    results = splunk_engine.get_matches(count_search, limit=1)
    if results:
        return int(results[0].get("count", 0))
    return 0


def _get_matched_raws(base_spl: str, splunk_engine: SplunkEngine, limit: int = 10000) -> Set[str]:
    raw_search = _strip_aggregation(base_spl) + " | table _raw"
    results = splunk_engine.get_matches(raw_search, limit=limit)
    matched = set()
    for r in results:
        raw_val = r.get("_raw", "")
        if isinstance(raw_val, list):
            matched.update(v for v in raw_val if v)
        elif raw_val:
            matched.add(raw_val)
    return matched


def evaluate_file(
    file_path: str,
    api_key: str,
    splunk_engine: SplunkEngine,
    tn_index: str,
    sourcetype: Optional[str] = None,
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
    print(f"\n{Colors.HEADER}=== Evaluating {os.path.basename(file_path)} ==={Colors.ENDC}")

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

    sourcetype_api_calls = 0
    if not sourcetype:
        print(f"{Colors.OKBLUE}[INFO] Sourcetype not provided. Inferring via LLM...{Colors.ENDC}")
        harness.reset_api_call_count()
        sourcetype = harness.infer_sourcetype(true_positives[0], api_key, vertexai=vertexai, vertex_project=vertex_project)
        sourcetype_api_calls = harness.get_api_call_count()
        print(f"{Colors.OKGREEN}[INFO] Inferred Sourcetype: {sourcetype}{Colors.ENDC}")

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
        tn_train = partition.tn_train
        test_count = partition.test_count
        run_idx = partition.run_idx

        log_run_header(partition, held_out_tps)

        uid = uuid.uuid4().hex[:8]
        GT_INDEX = f"gt_{uid}"
        eval_index = f"tp_{uid}"
        DISCOVERY_INDEX = f"disc_{uid}"

        setup_evaluation_index(GT_INDEX, splunk_engine)
        splunk_engine.ingest_line(seed_tp, GT_INDEX, sourcetype, "eval_host")

        setup_evaluation_index(DISCOVERY_INDEX, splunk_engine)
        splunk_engine.ingest_lines(partition.discovery_logs, DISCOVERY_INDEX, sourcetype, "eval_host")

        time.sleep(1)

        harness.reset_api_call_count()
        validated_rule, status = harness.generate_rule_for_log(
            seed_tp, GT_INDEX, api_key,
            splunk_engine=splunk_engine,
            description=description,
            vertexai=vertexai,
            vertex_project=vertex_project,
            verbose_llm=verbose_llm,
            main_dataset=DISCOVERY_INDEX,
            no_description=no_description,
        )
        api_calls = harness.get_api_call_count() + (sourcetype_api_calls if run_idx == 0 else 0)

        if not validated_rule:
            splunk_engine.delete_store(GT_INDEX)
            splunk_engine.delete_store(DISCOVERY_INDEX)
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
                "total_tns": len(tn_train),
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

        if held_out_tps:
            setup_evaluation_index(eval_index, splunk_engine)
            splunk_engine.ingest_lines(held_out_tps, eval_index, sourcetype, "eval_host", batch_size=1)
            time.sleep(3)

            tp_spl = _swap_index(validated_rule, GT_INDEX, eval_index)
            tp_hits = _count_raw_events(tp_spl, splunk_engine)
            matched_tp_raws = _get_matched_raws(tp_spl, splunk_engine)
            stripped_tp_raws = {r.strip() for r in matched_tp_raws}
            unmatched = [tp for tp in held_out_tps if tp.strip() not in stripped_tp_raws]
            recall = min(tp_hits / len(held_out_tps), 1.0)
            recall_display = f"{recall:.2%}"
            splunk_engine.delete_store(eval_index)
        else:
            tp_hits = 0
            unmatched = []
            recall = None
            recall_display = "N/A"

        print(f"{Colors.OKGREEN}[RESULT] Recall: {recall_display} ({tp_hits}/{len(held_out_tps)}){Colors.ENDC}")

        if global_tn:
            tn_spl = _swap_index(validated_rule, GT_INDEX, tn_index)
            matched_tn_raws = _get_matched_raws(tn_spl, splunk_engine)
            stripped_matched_tn_raws = {r.strip() for r in matched_tn_raws}
            all_global_tns = _get_matched_raws(f"index={tn_index}", splunk_engine)
            stripped_global_tns = {t.strip() for t in all_global_tns}
            matched_tns = [tn for tn in stripped_global_tns if tn in stripped_matched_tn_raws]
            total_tns = len(stripped_global_tns)
        else:
            tn_train_index = f"tn_train_{uuid.uuid4().hex[:8]}"
            setup_evaluation_index(tn_train_index, splunk_engine)
            splunk_engine.ingest_lines(tn_train, tn_train_index, sourcetype, "eval_host", batch_size=1)
            time.sleep(2)
            tn_spl_train = _swap_index(validated_rule, GT_INDEX, tn_train_index)
            matched_tn_raws = _get_matched_raws(tn_spl_train, splunk_engine)
            stripped_matched_tn_raws = {r.strip() for r in matched_tn_raws}
            matched_tns = [tn for tn in tn_train if tn.strip() in stripped_matched_tn_raws]
            total_tns = len(tn_train)
            splunk_engine.delete_store(tn_train_index)

        fp_hits = len(matched_tns)
        fpr = (fp_hits / total_tns) if total_tns > 0 else None
        print(f"{Colors.WARNING if fp_hits > 0 else Colors.OKGREEN}[RESULT] False Positives: {fp_hits}/{total_tns}{Colors.ENDC}")

        splunk_engine.delete_store(GT_INDEX)
        splunk_engine.delete_store(DISCOVERY_INDEX)

        result_entry = {
            **run_result_meta(partition),
            "status": "Success",
            "final_rule": validated_rule,
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
            "api_calls": api_calls,
        }

        if test_split is not None:
            if tp_test:
                test_eval_index = f"tp_test_{uuid.uuid4().hex[:8]}"
                setup_evaluation_index(test_eval_index, splunk_engine)
                splunk_engine.ingest_lines(tp_test, test_eval_index, sourcetype, "eval_host", batch_size=1)
                time.sleep(2)

                tp_spl_test = _swap_index(validated_rule, GT_INDEX, test_eval_index)
                tp_test_hits = _count_raw_events(tp_spl_test, splunk_engine)
                matched_tp_test_raws = _get_matched_raws(tp_spl_test, splunk_engine)
                stripped_tp_test_raws = {r.strip() for r in matched_tp_test_raws}
                test_unmatched = [tp for tp in tp_test if tp.strip() not in stripped_tp_test_raws]
                tp_test_recall = tp_test_hits / len(tp_test) if len(tp_test) > 0 else None
                splunk_engine.delete_store(test_eval_index)
            else:
                tp_test_hits = 0
                test_unmatched = []
                tp_test_recall = None

            if tn_test:
                tn_test_index = f"tn_test_{uuid.uuid4().hex[:8]}"
                setup_evaluation_index(tn_test_index, splunk_engine)
                splunk_engine.ingest_lines(tn_test, tn_test_index, sourcetype, "eval_host", batch_size=1)
                time.sleep(2)
                tn_spl_test = _swap_index(validated_rule, GT_INDEX, tn_test_index)
                matched_tn_test_raws = _get_matched_raws(tn_spl_test, splunk_engine)
                stripped_matched_tn_test_raws = {r.strip() for r in matched_tn_test_raws}
                matched_tn_test = [tn for tn in tn_test if tn.strip() in stripped_matched_tn_test_raws]
                tn_test_fp_hits = len(matched_tn_test)
                tn_test_total = len(tn_test)
                tn_test_fpr = tn_test_fp_hits / tn_test_total if tn_test_total > 0 else None
                splunk_engine.delete_store(tn_test_index)
            else:
                tn_test_fp_hits = 0
                tn_test_total = 0
                tn_test_fpr = None

            result_entry.update({
                "test_tp_hits": tp_test_hits,
                "test_tpr": tp_test_recall,
                "test_unmatched_tps": test_unmatched,
                "test_fp_hits": tn_test_fp_hits,
                "test_total_tns": tn_test_total,
                "test_fpr": tn_test_fpr,
                "test_total_tps": len(tp_test),
            })

        per_seed_results.append(result_entry)

    successful = [r for r in per_seed_results if r.get("status") == "Success"]
    measurable = [r for r in successful if r.get("tpr") is not None]
    avg_recall = (sum(r["tpr"] for r in measurable) / len(measurable)) if measurable else None
    total_tp_hits = sum(r["tp_hits"] for r in successful)
    total_held_out = sum(r["total_tps"] for r in successful)
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
