import argparse
import json
import os
import secrets
import sys
import time
from typing import List, Dict, Any, Optional, Set
import re
import contextlib
import agent as harness
from detection import (
    SplunkEngine, RegexEngine, PythonEngine, Colors
)
from eval.regex import setup_tn_index as _setup_tn_index_regex, evaluate_file as _evaluate_file_regex
from eval.python import setup_tn_index as _setup_tn_index_python, evaluate_file as _evaluate_file_python
from eval.splunk import setup_tn_index as _setup_tn_index_splunk, evaluate_file as _evaluate_file_splunk


# Vertex AI project id. Previously a hardcoded lab project; read from the
# environment so no cloud account id ships in the public artifact. Only used
# when --vertexai is passed.
_DEFAULT_VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")

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
    vertex_project: str = _DEFAULT_VERTEX_PROJECT,
    verbose_llm: bool = False,
    global_tn: bool = False,
    no_description: bool = False,
    cross_val: bool = False,
    N: int = 10,
    master_seed: int = 0,
) -> Dict[str, Any]:
    if mode == "regex":
        return _evaluate_file_regex(
            file_path, api_key, engine,
            vertexai=vertexai, vertex_project=vertex_project,
            verbose_llm=verbose_llm, global_tn=global_tn,
            no_description=no_description,
            cross_val=cross_val, N=N, master_seed=master_seed,
        )
    elif mode == "python":
        return _evaluate_file_python(
            file_path, api_key, engine,
            vertexai=vertexai, vertex_project=vertex_project,
            verbose_llm=verbose_llm, global_tn=global_tn,
            no_description=no_description,
            cross_val=cross_val, N=N, master_seed=master_seed,
        )
    elif mode == "splunk":
        if tn_index is None:
            raise ValueError("tn_index is required for splunk mode")
        return _evaluate_file_splunk(
            file_path, api_key, engine, tn_index,
            sourcetype=sourcetype,
            vertexai=vertexai, vertex_project=vertex_project,
            verbose_llm=verbose_llm, global_tn=global_tn,
            no_description=no_description,
            cross_val=cross_val, N=N, master_seed=master_seed,
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


def _build_summary(all_results: List[Dict[str, Any]], files_to_process: List[str], mode: str, global_tn: bool = False, master_seed: Optional[int] = None) -> Dict[str, Any]:
    summary_details = []
    total_matched_tns = 0
    total_unmatched_tps = 0
    total_api_calls = 0

    valid_tprs = []
    valid_fprs = []

    for r in all_results:
        # Build entry for 'details' list
        if r.get("status") != "Success":
            summary_details.append(r)
            continue
            
        metrics = r.get("metrics", {})
        per_seed_results = r.get("per_seed_results", [])
        
        successful_seeds = [seed for seed in per_seed_results if seed.get("status") == "Success"]
        trial_total_tns = sum(seed.get("total_tns", 0) for seed in successful_seeds)

        recall = metrics.get("recall")
        if metrics.get("total_tps", 0) > 1 and recall is not None:
            valid_tprs.append(recall)
            
        if trial_total_tns > 0:
            total_trial_fps = sum(seed.get("fp_hits", 0) for seed in successful_seeds)
            trial_fpr = total_trial_fps / trial_total_tns
            valid_fprs.append(trial_fpr)

        trials = r.get("trials", [])
        total_api_calls += sum(trial.get("api_calls", 0) for trial in trials)

        entry = {
            "file": r.get("file"),
            "status": r.get("status"),
            "recall": metrics.get("recall"),
            "avg_fp": metrics.get("avg_fp_hits"),
            "trials": trials,
        }
        
        tn_count = sum(seed.get("fp_hits", 0) for seed in successful_seeds)
        tp_miss_count = len(r.get("false_negatives", []))
        
        summary_details.append(entry)
        total_matched_tns += tn_count
        total_unmatched_tps += tp_miss_count

    successful = [r for r in all_results if r.get("status") == "Success"]
    
    avg_tpr = sum(valid_tprs) / len(valid_tprs) if valid_tprs else 0.0
    avg_fpr = sum(valid_fprs) / len(valid_fprs) if valid_fprs else 0.0
    
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "model": harness.MODEL_NAME,
        "master_seed": master_seed,
        "is_global_tn": global_tn,
        "total_files": len(files_to_process),
        "successful_generations": len(successful),
        "failed_generations": len(all_results) - len(successful),
        "average_true_positive_rate": avg_tpr,
        "average_false_positive_rate": avg_fpr,
        "total_false_positives": total_matched_tns,
        "total_false_negatives": total_unmatched_tps,
        "total_api_calls": total_api_calls,
        "details": summary_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Rule Generation Performance")
    
    # Input group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="Path to a single JSON dataset file")
    input_group.add_argument("--dir", help="Directory containing JSON dataset files")

    # Mode group
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--splunk-mode", action="store_true",
                           help="Use Splunk/SPL generation and testing (requires Splunk instance)")
    mode_group.add_argument("--regex-mode", action="store_true",
                           help="Use regex matching instead of Splunk/SPL (no Splunk instance needed)")
    mode_group.add_argument("--python-mode", action="store_true",
                           help="Use Python script matching inside Docker (no Splunk instance needed)")

    parser.add_argument("--output-dir", default="reports",
                        help="Directory for evaluation reports (default: reports/). Must not already exist.")
    parser.add_argument("--sourcetype", help="Sourcetype for logs (Splunk mode only, LLM inferred if not provided)")
    parser.add_argument("--splunk-url", default="https://localhost:8089")
    parser.add_argument("--splunk-user", default="admin")
    parser.add_argument("--splunk-pass", default="password")
    parser.add_argument("--gemini-api-key", help="Gemini API Key")
    parser.add_argument("--vertexai", action="store_true", help="Use Google Vertex AI instead of standard API call")
    parser.add_argument("--vertexai-project", default=_DEFAULT_VERTEX_PROJECT,
                        help="Vertex AI Project ID (default: $VERTEX_PROJECT)")
    parser.add_argument("--verbose-llm", action="store_true", help="Print raw LLM output and all MCP tool calls to stdout")
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
        "--N",
        type=int,
        default=10,
        help="Max number of cross-validation seeds to test per file (default: 10)."
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

    args = parser.parse_args()

    master_seed = args.seed if args.seed is not None else secrets.randbits(64)
    seed_origin = "" if args.seed is not None else " (random; pass --seed to reproduce)"
    print(f"{Colors.OKBLUE}[INFO] master_seed = {master_seed}{seed_origin}{Colors.ENDC}")

    # Validate mode-specific requirements
    if args.splunk_mode and not args.tn_index_name:
        print(f"{Colors.FAIL}Error: --tn-index-name is required in Splunk mode.{Colors.ENDC}")
        sys.exit(1)
 
    api_key = _resolve_api_key(args) if not args.vertexai else ""
    files_to_process = _collect_files(args)

    output_dir = args.output_dir
    if os.path.exists(output_dir) and not args.resume:
        print(f"{Colors.FAIL}Error: Output directory '{output_dir}' already exists. "
              f"Remove it, choose a different --output-dir, or pass --resume.{Colors.ENDC}")
        sys.exit(1)

    files_dir = os.path.join(output_dir, "files")
    os.makedirs(files_dir, exist_ok=True)

    pending_files = _filter_pending_files(files_to_process, files_dir) if args.resume else files_to_process
    if args.resume:
        skipped = len(files_to_process) - len(pending_files)
        if skipped:
            print(f"{Colors.OKBLUE}[INFO] Resume: skipping {skipped} file(s) with existing reports in {files_dir}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}[INFO] Resume: {len(pending_files)} file(s) remaining to evaluate{Colors.ENDC}")

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
                    vertexai=args.vertexai, vertex_project=args.vertexai_project,
                    verbose_llm=args.verbose_llm, global_tn=args.global_tn,
                    no_description=args.no_description,
                    cross_val=args.cross_val, N=args.N,
                    master_seed=master_seed,
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
                    vertexai=args.vertexai, vertex_project=args.vertexai_project,
                    verbose_llm=args.verbose_llm, global_tn=args.global_tn,
                    no_description=args.no_description,
                    cross_val=args.cross_val, N=args.N,
                    master_seed=master_seed,
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
                    tn_index=tn_index, sourcetype=args.sourcetype,
                    vertexai=args.vertexai, vertex_project=args.vertexai_project,
                    verbose_llm=args.verbose_llm,
                    global_tn=args.global_tn,
                    no_description=args.no_description,
                    cross_val=args.cross_val, N=args.N,
                    master_seed=master_seed,
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
    elif args.splunk_mode:
        mode = "splunk"
    if args.resume:
        all_results = _load_results_from_output_dir(files_dir)
    summary = _build_summary(all_results, files_to_process, mode, global_tn=args.global_tn, master_seed=master_seed)

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
