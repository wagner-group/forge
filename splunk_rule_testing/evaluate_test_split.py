import argparse
import json
import os
import secrets
import sys
import time
from typing import List, Dict, Any, Optional, Set, Tuple
import re
import contextlib
import agent as harness
from detection import (
    SplunkEngine, RegexEngine, PythonEngine, Colors
)
from eval_test_split.regex import setup_tn_index as _setup_tn_index_regex, evaluate_file as _evaluate_file_regex
from eval_test_split.python import setup_tn_index as _setup_tn_index_python, evaluate_file as _evaluate_file_python
from eval_test_split.splunk import setup_tn_index as _setup_tn_index_splunk, evaluate_file as _evaluate_file_splunk
from eval_test_split.claude import evaluate_file as _evaluate_file_claude
from eval_test_split._splitting import (
    RuleSplit,
    load_rule_slots,
    build_splits_for_run,
    build_splits_by_run,
    rule_splits_for_file,
    use_global_split_mode,
    run_seed,
    compute_batch_seeds,
    GLOBAL_DATASET_SIZE,
)
from claude.agent import check_claude_prerequisites

class Tee:
    def __init__(self, *files):
        self.files = files
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, obj):
        for f in self.files:
            # If it's a file object (not sys.stdout), strip ANSI
            if hasattr(f, 'name') and f.name != '<stdout>' and f.name != '<stderr>':
                clean_obj = self.ansi_escape.sub('', obj)
                f.write(clean_obj)
            else:
                f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

@contextlib.contextmanager
def LoggerContext(log_path: str):
    with open(log_path, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        try:
            yield
        finally:
            sys.stdout = original_stdout

# Mode dispatch
# ---------------------------------------------------------------------------

def setup_tn_index(
    mode: str,
    engine,
    files_to_process: List[str],
    tn_index: Optional[str] = None,
    sourcetype: Optional[str] = None,
    api_key: str = "",
) -> int:
    if mode == "regex":
        return _setup_tn_index_regex(engine, files_to_process)
    elif mode == "python":
        return _setup_tn_index_python(engine, files_to_process)
    elif mode == "splunk":
        if tn_index is None:
            raise ValueError("tn_index is required for splunk mode")
        return _setup_tn_index_splunk(tn_index, files_to_process, engine, sourcetype, api_key)
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_file(
    mode: str,
    file_path: str,
    api_key: str,
    engine,
    tn_index: Optional[str] = None,
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
    claude_bin: str = "claude",
    claude_model: str = "claude-sonnet-4-6",
    claude_dataset_root: str = "",
    workspaces_dir: str = "",
    claude_proxy_url: Optional[str] = None,
    claude_proxy_key: Optional[str] = None,
    claude_use_gemini_proxy: bool = True,
    rule_split_by_run: Optional[Dict[int, RuleSplit]] = None,
) -> Dict[str, Any]:
    if mode == "regex":
        return _evaluate_file_regex(
            file_path, api_key, engine,
            test_split=test_split, num_trial=num_trial,
            vertexai=vertexai, vertex_project=vertex_project,
            verbose_llm=verbose_llm, global_tn=global_tn,
            no_description=no_description,
            cross_val=cross_val, N=N, master_seed=master_seed,
            rule_split_by_run=rule_split_by_run,
        )
    elif mode == "python":
        return _evaluate_file_python(
            file_path, api_key, engine,
            test_split=test_split, num_trial=num_trial,
            vertexai=vertexai, vertex_project=vertex_project,
            verbose_llm=verbose_llm, global_tn=global_tn,
            no_description=no_description,
            cross_val=cross_val, N=N, master_seed=master_seed,
            rule_split_by_run=rule_split_by_run,
        )
    elif mode == "splunk":
        if tn_index is None:
            raise ValueError("tn_index is required for splunk mode")
        return _evaluate_file_splunk(
            file_path, api_key, engine, tn_index,
            test_split=test_split, num_trial=num_trial,
            sourcetype=sourcetype,
            vertexai=vertexai, vertex_project=vertex_project,
            verbose_llm=verbose_llm, global_tn=global_tn,
            no_description=no_description,
            cross_val=cross_val, N=N, master_seed=master_seed,
            rule_split_by_run=rule_split_by_run,
        )
    elif mode == "claude":
        return _evaluate_file_claude(
            file_path, api_key, engine,
            test_split=test_split, num_trial=num_trial,
            vertexai=vertexai, vertex_project=vertex_project,
            verbose_llm=verbose_llm, global_tn=global_tn,
            no_description=no_description,
            cross_val=cross_val, N=N, master_seed=master_seed,
            rule_split_by_run=rule_split_by_run,
            claude_bin=claude_bin,
            claude_model=claude_model,
            claude_dataset_root=claude_dataset_root,
            workspaces_dir=workspaces_dir,
            claude_proxy_url=claude_proxy_url,
            claude_proxy_key=claude_proxy_key,
            claude_use_gemini_proxy=claude_use_gemini_proxy,
        )
    raise ValueError(f"Unknown mode: {mode}")


def _resolve_api_key(args) -> str:
    """Return the Gemini API key from args or well-known file locations."""
    api_key = args.gemini_api_key
    if not api_key:
        possible_paths = [
            "gemini.txt",
            "../gemini.txt",
            "../../gemini.txt",
            os.path.join(os.path.dirname(__file__), "..", "gemini.txt"),
            os.path.join(os.path.dirname(__file__), "..", "..", "gemini.txt"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    api_key = f.read().strip()
                break
    if not api_key:
        print(f"{Colors.FAIL}Error: Gemini API Key not found.{Colors.ENDC}")
        sys.exit(1)
    return api_key


def _collect_files(args) -> List[str]:
    files_to_process: List[str] = []
    if args.file:
        files_to_process.append(args.file)
    else:
        for fname in os.listdir(args.dir):
            if fname.endswith(".json"):
                files_to_process.append(os.path.join(args.dir, fname))
    return files_to_process


def _report_path(files_dir: str, file_path: str) -> str:
    return os.path.join(files_dir, os.path.basename(file_path))


def _filter_pending_files(files_to_process: List[str], files_dir: str) -> List[str]:
    """Return input files that do not yet have a report JSON in files_dir."""
    return [f for f in files_to_process if not os.path.exists(_report_path(files_dir, f))]


def _prepare_test_split_plan(
    args,
    files_to_process: List[str],
    master_seed: int,
    output_dir: str,
) -> Tuple[Optional[Dict[int, Dict[str, RuleSplit]]], str]:
    """Precompute global or per-file-fallback splits; write run manifests."""
    if args.test_split is None:
        return None, ""

    is_dir = args.dir is not None
    global_mode = use_global_split_mode(
        test_split=args.test_split,
        dir_file_count=len(files_to_process),
        is_dir=is_dir,
    )
    mode_label = "global" if global_mode else "per_file"

    if global_mode:
        slots = load_rule_slots(files_to_process)
        splits_by_run: Dict[int, Dict[str, RuleSplit]] = {}
        for run_index in range(1, args.num_trial + 1):
            rs = run_seed(master_seed, run_index)
            by_name, manifest = build_splits_for_run(slots, rs, args.test_split)
            splits_by_run[run_index] = by_name
            manifest_path = os.path.join(output_dir, f"split_manifest_run_{run_index:02d}.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({**manifest, "run_index": run_index, "split_mode": mode_label}, f, indent=2)
        print(
            f"{Colors.OKBLUE}[INFO] Test-split mode: global ({GLOBAL_DATASET_SIZE} files), "
            f"{args.num_trial} run(s), run_seed = master_seed + run_index "
            f"(master_seed={master_seed}){Colors.ENDC}"
        )
        return splits_by_run, mode_label

    print(
        f"{Colors.OKBLUE}[INFO] Test-split mode: per-file fallback "
        f"({len(files_to_process)} file(s)), run_seed = master_seed + run_index "
        f"(master_seed={master_seed}){Colors.ENDC}"
    )
    return None, mode_label


def _rule_splits_for_file_path(
    file_path: str,
    *,
    global_splits_by_run: Optional[Dict[int, Dict[str, RuleSplit]]],
    master_seed: int,
    num_trial: int,
    test_split: float,
) -> Dict[int, RuleSplit]:
    """Return {run_index: RuleSplit} for one file."""
    file_name = os.path.basename(file_path)
    if global_splits_by_run is not None:
        return rule_splits_for_file(global_splits_by_run, file_name)

    slot = load_rule_slots([file_path])[0]
    per_file = build_splits_by_run(
        [slot],
        master_seed=master_seed,
        num_trial=num_trial,
        test_split=test_split,
    )
    return rule_splits_for_file(per_file, file_name)


def _load_results_from_output_dir(files_dir: str) -> List[Dict[str, Any]]:
    """Load all per-file report JSON objects from the output files directory."""
    results: List[Dict[str, Any]] = []
    if not os.path.isdir(files_dir):
        return results
    for fname in sorted(os.listdir(files_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(files_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"{Colors.WARNING}[WARN] Skipping unreadable report {path}: {e}{Colors.ENDC}")
    return results


def _build_summary(all_results: List[Dict[str, Any]], files_to_process: List[str], mode: str, global_tn: bool = False, master_seed: Optional[int] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    summary_details = []
    total_matched_tns = 0
    total_unmatched_tps = 0
    total_api_calls = 0
    total_tokens: Dict[str, int] = {}

    valid_tprs = []
    valid_fprs = []
    valid_test_tprs = []
    valid_test_fprs = []
    total_test_fps = 0
    total_test_fns = 0

    for r in all_results:
        # Build entry for 'details' list
        if r.get("status") != "Success":
            summary_details.append(r)
            continue
            
        metrics = r.get("metrics", {})
        per_seed_results = r.get("per_seed_results", [])
        
        successful_seeds = [seed for seed in per_seed_results if seed.get("status") == "Success"]
        trial_total_tns = sum(seed.get("total_tns", 0) for seed in successful_seeds)
        
        if metrics.get("total_tps", 0) > 1 and metrics.get("recall") is not None:
            valid_tprs.append(metrics.get("recall"))
        # collect per-file test metrics if present
        if metrics.get("average_test_true_positive_rate") is not None:
            valid_test_tprs.append(metrics.get("average_test_true_positive_rate"))
            
        if trial_total_tns > 0:
            total_trial_fps = sum(seed.get("fp_hits", 0) for seed in successful_seeds)
            trial_fpr = total_trial_fps / trial_total_tns
            valid_fprs.append(trial_fpr)
        if metrics.get("average_test_false_positive_rate") is not None:
            valid_test_fprs.append(metrics.get("average_test_false_positive_rate", 0))

        trials = r.get("trials", [])
        total_api_calls += sum(trial.get("api_calls", 0) for trial in trials)
        for trial in trials:
            tu = trial.get("token_usage") or {}
            for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
                total_tokens[k] = total_tokens.get(k, 0) + tu.get(k, 0)

        entry = {
            "file": r.get("file"),
            "status": r.get("status"),
            "recall": metrics.get("recall"),
            "avg_fp": metrics.get("avg_fp_hits"),
            "trials": trials,
        }
        
        tn_count = sum(seed.get("fp_hits", 0) for seed in successful_seeds)
        tp_miss_count = len(r.get("false_negatives", []))
        # test-level counts
        total_test_fps += metrics.get("total_test_false_positives", 0) or 0
        total_test_fns += metrics.get("total_test_false_negatives", 0) or 0
        
        summary_details.append(entry)
        total_matched_tns += tn_count
        total_unmatched_tps += tp_miss_count

    successful = [r for r in all_results if r.get("status") == "Success"]
    
    avg_tpr = sum(valid_tprs) / len(valid_tprs) if valid_tprs else 0.0
    avg_fpr = sum(valid_fprs) / len(valid_fprs) if valid_fprs else 0.0
    avg_test_tpr = sum(valid_test_tprs) / len(valid_test_tprs) if valid_test_tprs else None
    avg_test_fpr = sum(valid_test_fprs) / len(valid_test_fprs) if valid_test_fprs else None
    
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "model": model_name if model_name is not None else harness.MODEL_NAME,
        "master_seed": master_seed,
        "is_global_tn": global_tn,
        "total_files": len(files_to_process),
        "successful_generations": len(successful),
        "failed_generations": len(all_results) - len(successful),
        "average_true_positive_rate": avg_tpr,
        "average_false_positive_rate": avg_fpr,
        "average_test_true_positive_rate": avg_test_tpr,
        "average_test_false_positive_rate": avg_test_fpr,
        "total_false_positives": total_matched_tns,
        "total_false_negatives": total_unmatched_tps,
        "total_test_false_positives": total_test_fps,
        "total_test_false_negatives": total_test_fns,
        "total_api_calls": total_api_calls,
        "total_tokens": total_tokens,
        "details": summary_details,
    }


# Reference tree for --smoke-test-partition only. Lived on the original lab
# machine; set $PARTITION_REFERENCE_ROOT to point at your own copy.
_PARTITION_REFERENCE_ROOT_DEFAULT = os.environ.get("PARTITION_REFERENCE_ROOT", "")
_SMOKE_MANIFEST_KEYS = (
    "run_seed",
    "train_lines",
    "test_lines",
    "pool_lines_for_cut",
    "protected_seed_count",
)
_SMOKE_SPLIT_FIELDS = (
    ("true_positives_train", "tp_train"),
    ("true_positives_test", "tp_test"),
    ("true_negatives_train", "tn_train"),
    ("true_negatives_test", "tn_test"),
)
_SMOKE_MAX_ERRORS_SHOWN = 20


def _smoke_test_partition_scheme(
    *,
    dataset_dir: str,
    reference_root: str,
    master_seed: int,
    test_split: float,
    num_trial: int,
    N: int,
) -> List[str]:
    """Compare computed global splits and batch seeds against frozen reference trees."""
    errors: List[str] = []
    files_to_process = _collect_files_from_dir(dataset_dir)
    if len(files_to_process) != GLOBAL_DATASET_SIZE:
        errors.append(
            f"Expected {GLOBAL_DATASET_SIZE} JSON files in {dataset_dir}, "
            f"found {len(files_to_process)}"
        )
        return errors

    if not os.path.isdir(reference_root):
        errors.append(f"Reference root not found: {reference_root}")
        return errors

    slots = load_rule_slots(files_to_process)
    slot_by_name = {s.file_name: s for s in slots}

    for run_index in range(1, num_trial + 1):
        run_label = f"run_{run_index:02d}"
        run_root = os.path.join(reference_root, run_label)
        split_dir = os.path.join(run_root, "split_dataset")
        batch_dir = os.path.join(run_root, "eval", f"rep_{run_index:02d}_no_description")

        if not os.path.isdir(split_dir):
            errors.append(f"{run_label}: missing split_dataset/ under {reference_root}")
            continue
        if not os.path.isdir(batch_dir):
            errors.append(f"{run_label}: missing eval/rep_{run_index:02d}_no_description/")
            continue

        rs = run_seed(master_seed, run_index)
        by_name, manifest = build_splits_for_run(slots, rs, test_split)

        manifest_path = os.path.join(split_dir, "split_manifest.json")
        if not os.path.isfile(manifest_path):
            errors.append(f"{run_label}: missing split_manifest.json")
            continue
        with open(manifest_path, encoding="utf-8") as f:
            ref_manifest = json.load(f)
        for key in _SMOKE_MANIFEST_KEYS:
            expected = manifest.get(key) if key != "run_seed" else rs
            actual = ref_manifest.get(key)
            if actual != expected:
                errors.append(
                    f"{run_label}/split_manifest.json: {key} "
                    f"ref={actual!r} expected={expected!r}"
                )

        for fname, split in by_name.items():
            ref_path = os.path.join(split_dir, fname)
            if not os.path.isfile(ref_path):
                errors.append(f"{run_label}/{fname}: missing in reference split_dataset")
                continue
            with open(ref_path, encoding="utf-8") as f:
                ref = json.load(f)
            for ref_key, attr in _SMOKE_SPLIT_FIELDS:
                ours = getattr(split, attr)
                ref_lines = ref.get(ref_key, [])
                if set(ours) != set(ref_lines):
                    errors.append(
                        f"{run_label}/{fname}: {ref_key} mismatch "
                        f"(ours={len(ours)} ref={len(ref_lines)})"
                    )

            slot = slot_by_name[fname].slot
            slot_label = f"{slot:02d}"
            expected_seeds = compute_batch_seeds(split, rs, N)
            for batch_idx, seed_tp in enumerate(expected_seeds, start=1):
                batch_path = os.path.join(
                    batch_dir, f"batch_{batch_idx:03d}", f"{slot_label}.json"
                )
                if not os.path.isfile(batch_path):
                    errors.append(
                        f"{run_label}/{fname}: missing batch_{batch_idx:03d}/{slot_label}.json"
                    )
                    continue
                with open(batch_path, encoding="utf-8") as f:
                    batch_payload = json.load(f)
                ref_tp = batch_payload.get("true_positive")
                if ref_tp != seed_tp:
                    errors.append(
                        f"{run_label}/{fname}: batch_{batch_idx:03d} seed mismatch"
                    )
                    break

    return errors


def _collect_files_from_dir(dataset_dir: str) -> List[str]:
    return sorted(
        os.path.join(dataset_dir, fname)
        for fname in os.listdir(dataset_dir)
        if fname.endswith(".json")
    )


def _validate_smoke_test_args(args) -> Optional[str]:
    if args.file:
        return "--smoke-test-partition requires --dir (66 JSON files), not --file"
    if not args.dir:
        return "--smoke-test-partition requires --dir"
    if args.test_split is None:
        return "--smoke-test-partition requires --test-split"
    if args.num_trial is None or args.num_trial <= 0:
        return "--smoke-test-partition requires --num-trial > 0"
    if args.seed is None:
        return "--smoke-test-partition requires --seed"
    if not (0.0 < args.test_split < 1.0):
        return "--test-split must be a real number between 0 and 1"
    if args.N <= 0:
        return "--N must be > 0"
    return None


def _run_smoke_test_partition(args) -> int:
    err = _validate_smoke_test_args(args)
    if err:
        print(f"{Colors.FAIL}Error: {err}{Colors.ENDC}")
        return 1

    reference_root = args.partition_reference_root
    print(
        f"{Colors.OKBLUE}[INFO] Partition smoke test: "
        f"dataset={args.dir}, reference={reference_root}, "
        f"master_seed={args.seed}, test_split={args.test_split}, "
        f"num_trial={args.num_trial}, N={args.N}{Colors.ENDC}"
    )

    errors = _smoke_test_partition_scheme(
        dataset_dir=args.dir,
        reference_root=reference_root,
        master_seed=args.seed,
        test_split=args.test_split,
        num_trial=args.num_trial,
        N=args.N,
    )

    if errors:
        print(f"{Colors.FAIL}[FAIL] Partition smoke test: {len(errors)} error(s){Colors.ENDC}")
        for msg in errors[:_SMOKE_MAX_ERRORS_SHOWN]:
            print(f"  - {msg}")
        if len(errors) > _SMOKE_MAX_ERRORS_SHOWN:
            print(f"  ... and {len(errors) - _SMOKE_MAX_ERRORS_SHOWN} more")
        print(
            f"{Colors.WARNING}[HINT] Reference trees were built from "
            f"log_rules_dataset_final/dataset; dataset/final may differ "
            f"(e.g. linux_auditd_disable_or_modify_system_firewall.json).{Colors.ENDC}"
        )
        return 1

    print(
        f"{Colors.OKGREEN}[OK] Partition smoke test passed "
        f"({args.num_trial} runs, {GLOBAL_DATASET_SIZE} rules, batch seeds N={args.N})"
        f"{Colors.ENDC}"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Rule Generation Performance")
    
    # Input group (--dir required for eval or smoke-test-partition)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--file", help="Path to a single JSON dataset file")
    input_group.add_argument("--dir", help="Directory containing JSON dataset files")

    # Mode group (not required for --smoke-test-partition)
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--splunk-mode", action="store_true",
                           help="Use Splunk/SPL generation and testing (requires Splunk instance)")
    mode_group.add_argument("--regex-mode", action="store_true",
                           help="Use regex matching instead of Splunk/SPL (no Splunk instance needed)")
    mode_group.add_argument("--python-mode", action="store_true",
                           help="Use Python script matching inside Docker (no Splunk instance needed)")
    mode_group.add_argument("--claude-mode", action="store_true",
                           help="Use Claude Code CLI for regex generation (requires claude + auth or Gemini proxy)")

    parser.add_argument("--claude-bin", default="claude",
                        help="Path to Claude Code CLI binary (default: claude)")
    parser.add_argument("--claude-dataset-root",
                        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "log_rules_dataset_final")),
                        help="Root of log_rules_dataset_final for md_files and filename mapping")
    parser.add_argument("--claude-model", default="claude-sonnet-4-6",
                        help="Claude Code model alias (LiteLLM maps this to Gemini when using the proxy)")
    parser.add_argument("--claude-proxy-url", default=None,
                        help="LiteLLM proxy URL (default: auto-detect http://127.0.0.1:4000)")
    parser.add_argument("--claude-proxy-key", default=None,
                        help="LiteLLM master key for a local proxy (default: local-litellm-proxy-key)")
    parser.add_argument("--no-claude-gemini-proxy", action="store_true",
                        help="Do not auto-configure local Gemini LiteLLM proxy; require ANTHROPIC_API_KEY")

    parser.add_argument("--output-dir", default="reports",
                        help="Directory for evaluation reports (default: reports/). Must not already exist.")
    parser.add_argument("--sourcetype", help="Sourcetype for logs (Splunk mode only, LLM inferred if not provided)")
    parser.add_argument("--splunk-url", default="https://localhost:8089")
    parser.add_argument("--splunk-user", default="admin")
    parser.add_argument("--splunk-pass", default="password")
    parser.add_argument("--provider", choices=["gemini", "openai"], default=None,
                        help="LLM backend for FORGE (default: gemini, or $FORGE_PROVIDER)")
    parser.add_argument("--openai-model", default=None,
                        help="OpenAI model id when --provider openai (default: gpt-5.6-luna)")
    parser.add_argument("--openai-effort", default=None,
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="reasoning.effort; always sent explicitly (default: none)")
    parser.add_argument("--gemini-api-key", help="Gemini API Key")
    parser.add_argument("--vertexai", action="store_true", help="Use Google Vertex AI instead of standard API call")
    parser.add_argument("--vertexai-project", default=os.environ.get("VERTEX_PROJECT", ""),
                        help="Vertex AI Project ID (default: $VERTEX_PROJECT)")
    parser.add_argument(
        "--verbose-llm",
        action="store_true",
        help="Print raw LLM output to stdout (regex mode: MCP tool calls; claude mode: live stream-json thoughts/tools)",
    )
    parser.add_argument("--no-description", action="store_true", help="Do not pass the log description to the LLM")

    parser.add_argument(
        "--new-tn-index",
        action="store_true",
        help="Collect, deduplicate, and ingest all TNs into a fresh --tn-index-name before evaluation (Splunk mode only).",
    )
    parser.add_argument(
        "--tn-index-name",
        help="Name of the Splunk index used for True Negative evaluation (required in Splunk mode).",
    )
    parser.add_argument(
        "--global-tn",
        action="store_true",
        help="Detect False Positives against the entire TN dataset instead of just the local file lists."
    )
    parser.add_argument(
        "--cross-val",
        action="store_true",
        help="Iterate through all True Positives in the dataset (Cross-Validation). If off, only the first TP is used."
    )
    parser.add_argument(
        "--test-split",
        type=float,
        help="Fraction of pooled TP+TN lines (global or per-file) held out for test "
             "(real number between 0 and 1).",
    )
    parser.add_argument(
        "--num-trial",
        type=int,
        help="Number of partition runs per evaluation (run_seed = master_seed + run_index, "
             "1-based). Each run uses up to --N batch seeds per file. "
             "Required when --test-split is provided.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=10,
        help="Max batch seed TPs per partition per file when --test-split is used, "
             "or max cross-validation seeds per file with --cross-val (default: 10)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for sampling and shuffles. If omitted, a fresh 64-bit "
             "seed is generated each run and recorded in summary.json."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing --output-dir: skip inputs that already have a report "
             "in output-dir/files/, and rebuild summary.json from all reports there.",
    )
    parser.add_argument(
        "--smoke-test-partition",
        action="store_true",
        help="Verify global partition scheme against frozen reference trees; "
             "requires --dir, --test-split, --num-trial, and --seed (no eval mode needed).",
    )
    parser.add_argument(
        "--partition-reference-root",
        default=_PARTITION_REFERENCE_ROOT_DEFAULT,
        help="Root of claude_eval_testsplit reference run folders for --smoke-test-partition.",
    )

    args = parser.parse_args()

    if args.provider:
        harness.PROVIDER = args.provider
    if args.openai_model:
        harness.OPENAI_MODEL = args.openai_model
    if args.openai_effort:
        harness.OPENAI_EFFORT = args.openai_effort
    if harness.PROVIDER == "openai":
        harness.MODEL_NAME = harness.OPENAI_MODEL
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"{Colors.FAIL}Error: OPENAI_API_KEY is required with --provider openai.{Colors.ENDC}")
            sys.exit(1)

    if args.smoke_test_partition:
        sys.exit(_run_smoke_test_partition(args))

    if not args.file and not args.dir:
        print(f"{Colors.FAIL}Error: --file or --dir is required.{Colors.ENDC}")
        sys.exit(1)
    if not (args.splunk_mode or args.regex_mode or args.python_mode or args.claude_mode):
        print(
            f"{Colors.FAIL}Error: one of --splunk-mode, --regex-mode, "
            f"--python-mode, or --claude-mode is required.{Colors.ENDC}"
        )
        sys.exit(1)

    master_seed = args.seed if args.seed is not None else secrets.randbits(64)
    seed_origin = "" if args.seed is not None else " (random; pass --seed to reproduce)"
    print(f"{Colors.OKBLUE}[INFO] master_seed = {master_seed}{seed_origin}{Colors.ENDC}")

    # Validate mode-specific requirements
    if args.splunk_mode and not args.tn_index_name:
        print(f"{Colors.FAIL}Error: --tn-index-name is required in Splunk mode.{Colors.ENDC}")
        sys.exit(1)
    # Validate test-split and cross-val mutual exclusion
    if args.test_split is not None and args.cross_val:
        print(f"{Colors.FAIL}Error: --test-split and --cross-val are mutually exclusive.{Colors.ENDC}")
        sys.exit(1)
    if args.test_split is not None and (args.num_trial is None or args.num_trial <= 0):
        print(f"{Colors.FAIL}Error: --num-trial is required and must be > 0 when --test-split is provided.{Colors.ENDC}")
        sys.exit(1)
    if args.test_split is not None and not (0.0 < args.test_split < 1.0):
        print(f"{Colors.FAIL}Error: --test-split must be a real number between 0 and 1.{Colors.ENDC}")
        sys.exit(1)

    if args.claude_mode:
        prereq_err = check_claude_prerequisites(
            args.claude_bin,
            proxy_url=args.claude_proxy_url,
            proxy_key=args.claude_proxy_key,
            use_gemini_proxy=not args.no_claude_gemini_proxy,
        )
        if prereq_err:
            print(f"{Colors.FAIL}Error: {prereq_err}{Colors.ENDC}")
            sys.exit(1)

    api_key = ""
    # The OpenAI backend reads OPENAI_API_KEY from the environment (see llm_openai),
    # so no Gemini key is needed -- and demanding one would block every Luna run.
    if not args.claude_mode and harness.PROVIDER != "openai":
        api_key = _resolve_api_key(args) if not args.vertexai else ""
    files_to_process = _collect_files(args)

    output_dir = args.output_dir
    if os.path.exists(output_dir) and not args.resume:
        print(f"{Colors.FAIL}Error: Output directory '{output_dir}' already exists. "
              f"Remove it, choose a different --output-dir, or pass --resume.{Colors.ENDC}")
        sys.exit(1)

    files_dir = os.path.join(output_dir, "files")
    workspaces_dir = os.path.join(output_dir, "workspaces")
    os.makedirs(files_dir, exist_ok=True)
    if args.claude_mode:
        os.makedirs(workspaces_dir, exist_ok=True)

    pending_files = _filter_pending_files(files_to_process, files_dir) if args.resume else files_to_process
    if args.resume:
        skipped = len(files_to_process) - len(pending_files)
        if skipped:
            print(f"{Colors.OKBLUE}[INFO] Resume: skipping {skipped} file(s) with existing reports in {files_dir}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}[INFO] Resume: {len(pending_files)} file(s) remaining to evaluate{Colors.ENDC}")

    global_splits_by_run, _split_mode = _prepare_test_split_plan(
        args, files_to_process, master_seed, output_dir
    )

    def _eval_kwargs(file_path: str) -> Dict[str, Any]:
        kw: Dict[str, Any] = {}
        if args.test_split is not None:
            kw["rule_split_by_run"] = _rule_splits_for_file_path(
                file_path,
                global_splits_by_run=global_splits_by_run,
                master_seed=master_seed,
                num_trial=args.num_trial,
                test_split=args.test_split,
            )
        return kw

    all_results: List[Dict[str, Any]] = []

    if args.regex_mode:
        # ---- Regex evaluation path ----
        engine = RegexEngine()
        setup_tn_index("regex", engine, files_to_process)

        for file_path in pending_files:
            base_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            log_path = os.path.join(files_dir, f"{base_name_no_ext}.log")
            
            with LoggerContext(log_path):
                result = evaluate_file(
                    "regex",
                    file_path, api_key, engine,
                    test_split=args.test_split, num_trial=args.num_trial,
                    vertexai=args.vertexai, vertex_project=args.vertexai_project,
                    verbose_llm=args.verbose_llm, global_tn=args.global_tn,
                    no_description=args.no_description,
                    cross_val=args.cross_val, N=args.N, master_seed=master_seed,
                    **_eval_kwargs(file_path),
                )
            
            all_results.append(result)

            base_name = os.path.basename(file_path)
            report_path = os.path.join(files_dir, base_name)
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"{Colors.OKBLUE}[INFO] Individual report saved to {report_path}{Colors.ENDC}")
    elif args.python_mode:
        # ---- Python evaluation path ----
        engine = PythonEngine()
        setup_tn_index("python", engine, files_to_process)
 
        for file_path in pending_files:
            base_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            log_path = os.path.join(files_dir, f"{base_name_no_ext}.log")

            with LoggerContext(log_path):
                result = evaluate_file(
                    "python",
                    file_path, api_key, engine,
                    test_split=args.test_split, num_trial=args.num_trial,
                    vertexai=args.vertexai, vertex_project=args.vertexai_project,
                    verbose_llm=args.verbose_llm, global_tn=args.global_tn,
                    no_description=args.no_description,
                    cross_val=args.cross_val, N=args.N, master_seed=master_seed,
                    **_eval_kwargs(file_path),
                )
            
            all_results.append(result)
 
            base_name = os.path.basename(file_path)
            report_path = os.path.join(files_dir, base_name)
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"{Colors.OKBLUE}[INFO] Individual report saved to {report_path}{Colors.ENDC}")
    elif args.claude_mode:
        engine = RegexEngine()
        setup_tn_index("regex", engine, files_to_process)

        for file_path in pending_files:
            base_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            log_path = os.path.join(files_dir, f"{base_name_no_ext}.log")

            with LoggerContext(log_path):
                result = evaluate_file(
                    "claude",
                    file_path, api_key, engine,
                    test_split=args.test_split, num_trial=args.num_trial,
                    verbose_llm=args.verbose_llm, global_tn=args.global_tn,
                    no_description=True,
                    cross_val=args.cross_val, N=args.N, master_seed=master_seed,
                    **_eval_kwargs(file_path),
                    claude_bin=args.claude_bin,
                    claude_model=args.claude_model,
                    claude_dataset_root=args.claude_dataset_root,
                    workspaces_dir=workspaces_dir,
                    claude_proxy_url=args.claude_proxy_url,
                    claude_proxy_key=args.claude_proxy_key,
                    claude_use_gemini_proxy=not args.no_claude_gemini_proxy,
                )

            all_results.append(result)

            base_name = os.path.basename(file_path)
            report_path = os.path.join(files_dir, base_name)
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"{Colors.OKBLUE}[INFO] Individual report saved to {report_path}{Colors.ENDC}")
    elif args.splunk_mode:
        # ---- Splunk evaluation path ----
        splunk_engine = SplunkEngine(args.splunk_url, args.splunk_user, args.splunk_pass)
        if args.new_tn_index:
            setup_tn_index(
                "splunk",
                splunk_engine,
                files_to_process,
                tn_index=args.tn_index_name,
                sourcetype=args.sourcetype,
                api_key=api_key,
            )
        tn_index = args.tn_index_name

        for file_path in pending_files:
            base_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            log_path = os.path.join(files_dir, f"{base_name_no_ext}.log")

            with LoggerContext(log_path):
                result = evaluate_file(
                    "splunk",
                    file_path, api_key,
                    splunk_engine,
                    test_split=args.test_split, num_trial=args.num_trial,
                    tn_index=tn_index, sourcetype=args.sourcetype,
                    vertexai=args.vertexai, vertex_project=args.vertexai_project,
                    verbose_llm=args.verbose_llm,
                    global_tn=args.global_tn,
                    no_description=args.no_description,
                    cross_val=args.cross_val, N=args.N, master_seed=master_seed,
                    **_eval_kwargs(file_path),
                )
            all_results.append(result)

            base_name = os.path.basename(file_path)
            report_path = os.path.join(files_dir, base_name)
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"{Colors.OKBLUE}[INFO] Individual report saved to {report_path}{Colors.ENDC}")
 
    if args.python_mode:
        mode = "python"
    elif args.regex_mode:
        mode = "regex"
    elif args.claude_mode:
        mode = "claude"
    elif args.splunk_mode:
        mode = "splunk"
    model_name = "claude-code" if args.claude_mode else harness.MODEL_NAME
    if args.resume:
        all_results = _load_results_from_output_dir(files_dir)
    summary = _build_summary(all_results, files_to_process, mode, global_tn=args.global_tn, master_seed=master_seed, model_name=model_name)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{Colors.HEADER}=== Evaluation Summary ==={Colors.ENDC}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Successful Generations: {summary['successful_generations']}")
    print(f"Total False Positives ({'Global' if summary.get('is_global_tn') else 'Local'}): {summary['total_false_positives']}")
    print(f"Total False Negatives: {summary['total_false_negatives']}")
    print(f"Total API Calls: {summary['total_api_calls']}")
    print(f"{Colors.OKGREEN}[INFO] Summary report saved to {summary_path}{Colors.ENDC}")


if __name__ == "__main__":
    main()
