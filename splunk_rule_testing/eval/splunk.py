import json
import os
import random
import time
import uuid
from typing import List, Dict, Any, Set, Optional
import agent as harness
from detection import SplunkEngine, Colors
from eval._seeding import derive_seed


# Vertex AI project id. Previously a hardcoded lab project; read from the
# environment so no cloud account id ships in the public artifact. Only used
# when --vertexai is passed.
_DEFAULT_VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")


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
    vertex_project: str = _DEFAULT_VERTEX_PROJECT,
    verbose_llm: bool = False,
    global_tn: bool = False,
    no_description: bool = False,
    cross_val: bool = False,
    N: int = 10,
    master_seed: int = 0,
) -> Dict[str, Any]:
    print(f"\n{Colors.HEADER}=== Evaluating {os.path.basename(file_path)} ==={Colors.ENDC}")

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

    sourcetype_api_calls = 0
    if not sourcetype:
        print(f"{Colors.OKBLUE}[INFO] Sourcetype not provided. Inferring via LLM...{Colors.ENDC}")
        harness.reset_api_call_count()
        sourcetype = harness.infer_sourcetype(true_positives[0], api_key, vertexai=vertexai, vertex_project=vertex_project)
        sourcetype_api_calls = harness.get_api_call_count()
        print(f"{Colors.OKGREEN}[INFO] Inferred Sourcetype: {sourcetype}{Colors.ENDC}")

    per_seed_results = []

    for seed_idx, seed_tp in enumerate(seeds_to_test):
        # held_out_tps = [tp for j, tp in enumerate(true_positives) if j != seed_idx]
        held_out_tps = [tp for tp in true_positives if tp != seed_tp]
        print(f"\n{Colors.OKBLUE}[RUN {seed_idx+1}/{len(seeds_to_test)}] Seed TP {seed_idx} "
              f"({len(held_out_tps)} held-out TPs){Colors.ENDC}")

        uid = uuid.uuid4().hex[:8]
        GT_INDEX = f"gt_{uid}"
        eval_index = f"tp_{uid}"
        DISCOVERY_INDEX = f"disc_{uid}"

        setup_evaluation_index(GT_INDEX, splunk_engine)
        splunk_engine.ingest_line(seed_tp, GT_INDEX, sourcetype, "eval_host")

        setup_evaluation_index(DISCOVERY_INDEX, splunk_engine)
        # Flat, unlabeled, byte-deduped, deterministically shuffled discovery pool.
        # Cross-file dups remain in the dataset by design; within-file dups were removed
        # at dataset level; collapse any residual TP/TN overlap and remove ingest-order bias.
        # (Splunk normally sorts by _time on query, but shuffle is belt-and-suspenders.)
        all_file_logs = list(dict.fromkeys(true_positives + true_negatives))
        shuffle_seed = derive_seed(master_seed, "shuffle", file_name, seed_idx)
        random.Random(shuffle_seed).shuffle(all_file_logs)
        splunk_engine.ingest_lines(all_file_logs, DISCOVERY_INDEX, sourcetype, "eval_host")

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
        api_calls = harness.get_api_call_count() + (sourcetype_api_calls if seed_idx == 0 else 0)

        if not validated_rule:
            splunk_engine.delete_store(GT_INDEX)
            per_seed_results.append({
                "seed_index": seed_idx,
                "status": f"Generation Failed: {status}",
                "recall": 0, "tp_hits": 0, "held_out_tps": len(held_out_tps),
                "fp_hits": 0,
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
            recall = None  # held-out is empty (e.g. single-TP file) — recall is not measurable
            recall_display = "N/A"

        print(f"{Colors.OKGREEN}[RESULT] Recall: {recall_display} ({tp_hits}/{len(held_out_tps)}){Colors.ENDC}")

        tn_spl = _swap_index(validated_rule, GT_INDEX, tn_index)
        matched_tn_raws = _get_matched_raws(tn_spl, splunk_engine)
        stripped_matched_tn_raws = {r.strip() for r in matched_tn_raws}

        if global_tn:
            all_global_tns = _get_matched_raws(f"index={tn_index}", splunk_engine)
            stripped_global_tns = {t.strip() for t in all_global_tns}
            matched_tns = [tn for tn in stripped_global_tns if tn in stripped_matched_tn_raws]
            total_tns = len(stripped_global_tns)
        else:
            matched_tns = [tn for tn in true_negatives if tn.strip() in stripped_matched_tn_raws]
            total_tns = len(true_negatives)

        fp_hits = len(matched_tns)
        print(f"{Colors.WARNING if fp_hits > 0 else Colors.OKGREEN}[RESULT] False Positives: {fp_hits}/{total_tns}{Colors.ENDC}")

        splunk_engine.delete_store(GT_INDEX)
        splunk_engine.delete_store(DISCOVERY_INDEX)

        per_seed_results.append({
            "seed_index": seed_idx,
            "status": "Success",
            "final_spl": validated_rule,
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
                "rule": r.get("final_spl"),
                "api_calls": r.get("api_calls", 0),
            } for r in per_seed_results
        ],
        "per_seed_results": per_seed_results,
        "false_negatives": list(set(all_unmatched)),
    }
