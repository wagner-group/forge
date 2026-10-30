#!/usr/bin/env python3
"""Lightweight Codex-CLI wrapper for regex rule generation.

Mirrors the shape of claude/agent.py + claude/workspace.py so it can later slot
into eval_test_split/, but is runnable standalone for experiments:

    python codex_agent.py --file ../dataset/final/linux_auditd_dd_file_overwrite.json

Design notes
------------
* `codex exec` is the headless entry point: prompt in, result out, exit.
* `--output-schema` constrains the final message to a JSON Schema, so we never
  parse prose. This is cleaner than the Claude path, which had the agent write a
  result file and then recovered the regex from the workspace on failure.
* The workspace holds the seed log and the discovery pool as *files*, so the
  agent reads them with its own tools rather than us stuffing 10k tokens of
  Sysmon XML into the prompt. That is the substantive difference from FORGE and
  the thing the harness comparison is meant to measure.
* Auth: `codex login` (ChatGPT subscription, draws on the plan's 5-hour window)
  or an API key (per-token billing). `codex login status` reports which.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MODEL = "gpt-5.6-luna"

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "regex": {
            "type": "string",
            "description": "The final Python-compatible regular expression.",
        },
        "explanation": {
            "type": "string",
            "description": "One paragraph: what behaviour it matches and why it generalises.",
        },
    },
    "required": ["regex", "explanation"],
    "additionalProperties": False,
}

PROMPT = """\
You are an expert Detection Engineer. Write ONE Python-compatible regular
expression that detects the malicious behaviour shown in the seed log.

Files in your working directory:
  seed_log.txt   - a single suspicious log line. Your regex MUST match it.
  pool.jsonl     - one log line per row, a MIXTURE of malicious and benign logs
                   related to this behaviour. You are NOT told which is which.
{description_block}
Method:
1. Read seed_log.txt. Separate stable signatures (API names, flags, paths) from
   volatile data (timestamps, GUIDs, PIDs, hostnames). Never match volatile data.
2. Read pool.jsonl and test candidate patterns against it with your own tooling
   (python3 is available). Look for BOTH:
   - benign lines your pattern wrongly matches (false positives), and
   - malicious variants your pattern misses (false negatives).
3. The seed is ONE example. Aim for high recall on the underlying behaviour, not
   a literal match of this string. Prefer broad OR-style coverage over narrow
   AND-chains: every extra AND condition excludes some real attack.
4. Verify your final regex matches seed_log.txt before finishing.

Return the final regex in the required JSON output shape.
"""


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

def prepare_workspace(root: str, seed_tp: str, pool: List[str],
                      description: Optional[str] = None) -> str:
    """Materialise a per-trial workspace: seed log + discovery pool."""
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)

    with open(os.path.join(root, "seed_log.txt"), "w", encoding="utf-8") as f:
        f.write(seed_tp)

    with open(os.path.join(root, "pool.jsonl"), "w", encoding="utf-8") as f:
        for line in pool:
            f.write(json.dumps({"line": line}) + "\n")

    if description:
        with open(os.path.join(root, "description.txt"), "w", encoding="utf-8") as f:
            f.write(description)

    with open(os.path.join(root, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(RESULT_SCHEMA, f, indent=2)

    return root


def build_prompt(has_description: bool) -> str:
    block = ("  description.txt - a short description of the malicious behaviour. Read it.\n"
             if has_description else "")
    return PROMPT.format(description_block=block)


# ---------------------------------------------------------------------------
# Usage extraction from --json event stream
# ---------------------------------------------------------------------------

def parse_usage(jsonl_text: str) -> Dict[str, int]:
    """Best-effort token accounting from the JSONL event stream.

    The event schema is not contractually stable, so this scans for any object
    carrying token-ish counters and keeps the largest value seen for each
    (counters are typically cumulative per turn).
    """
    keys = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "cached_tokens": ("cached_input_tokens", "cached_tokens"),
        "reasoning_tokens": ("reasoning_output_tokens", "reasoning_tokens"),
    }
    out: Dict[str, int] = {k: 0 for k in keys}

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for canon, aliases in keys.items():
                for a in aliases:
                    v = o.get(a)
                    if isinstance(v, int) and v > out[canon]:
                        out[canon] = v
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            walk(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def extract_regex(last_message: str) -> Optional[str]:
    """Pull the regex out of the schema-constrained final message."""
    try:
        obj = json.loads(last_message)
        if isinstance(obj, dict) and isinstance(obj.get("regex"), str):
            return obj["regex"]
    except json.JSONDecodeError:
        pass
    # Fallback: a fenced or bare JSON object somewhere in the text.
    m = re.search(r'\{[^{}]*"regex"\s*:\s*"(?:[^"\\]|\\.)*"[^{}]*\}', last_message, re.S)
    if m:
        try:
            return json.loads(m.group(0)).get("regex")
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------

def run_codex_trial(
    workspace: str,
    *,
    model: str = DEFAULT_MODEL,
    has_description: bool = True,
    codex_bin: str = "codex",
    effort: str = "none",
    timeout: int = 900,
    verbose: bool = False,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Run one Codex trial. Returns (regex, status, meta)."""
    last_msg_path = os.path.join(workspace, "_last_message.txt")
    schema_path = os.path.join(workspace, "schema.json")

    cmd = [
        codex_bin, "exec",
        "--cd", workspace,
        "--model", model,
        # Explicit, and none by default: FORGE runs at effort none, so leaving
        # this at the CLI default made the head-to-head not comparable on cost.
        # Omitting it also bills ~1.5x more than sending "none" (PORT_DESIGN s1).
        "-c", f'model_reasoning_effort="{effort}"',
        "--sandbox", "workspace-write",
        "--skip-git-repo-check",
        "--output-schema", schema_path,
        "--output-last-message", last_msg_path,
        "--json",
        build_prompt(has_description),
    ]

    if verbose:
        print("[codex] " + " ".join(cmd[:-1]) + " <prompt>")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout}s", {"elapsed": timeout}

    elapsed = time.time() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    with open(os.path.join(workspace, "codex_run.log"), "w", encoding="utf-8") as f:
        f.write(combined)

    usage = parse_usage(proc.stdout or "")
    meta: Dict[str, Any] = {
        "elapsed_sec": round(elapsed, 1),
        "returncode": proc.returncode,
        "token_usage": usage,
        "workspace": workspace,
    }

    last_msg = ""
    if os.path.exists(last_msg_path):
        with open(last_msg_path, encoding="utf-8") as f:
            last_msg = f.read().strip()

    if proc.returncode != 0:
        tail = combined.strip().splitlines()[-5:]
        return None, f"codex exec failed (rc={proc.returncode}): {' | '.join(tail)}", meta

    rgx = extract_regex(last_msg)
    if not rgx:
        return None, f"Could not extract regex from final message ({len(last_msg)} chars)", meta
    return rgx, "Success", meta


# ---------------------------------------------------------------------------
# Standalone experiment entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="dataset/final/*.json")
    ap.add_argument("--seed-index", type=int, default=0, help="which TP to use as the seed")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="none",
                    choices=("none", "low", "medium", "high"),
                    help="model_reasoning_effort (default none, matching FORGE)")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--no-description", action="store_true")
    ap.add_argument("--pool-cap", type=int, default=0,
                    help="Cap pool.jsonl at N lines (0 = no cap). Use for cheap experiments.")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)
    tps = data.get("true_positives") or []
    tns = data.get("true_negatives") or []
    if not tps:
        print("No true_positives in that file.")
        return 1

    seed = tps[args.seed_index]
    held_out = [t for i, t in enumerate(tps) if i != args.seed_index]
    pool = held_out + tns
    if args.pool_cap:
        pool = pool[:args.pool_cap]

    base = os.path.splitext(os.path.basename(args.file))[0]
    ws = args.workspace or os.path.join("codex_workspaces", base)
    desc = None if args.no_description else (data.get("description") or None)

    prepare_workspace(ws, seed, pool, desc)
    print(f"[setup] workspace={ws}  seed={len(seed)} chars  pool={len(pool)} lines "
          f"(held-out TPs: {len(held_out)}, TNs: {len(tns)})")

    rgx, status, meta = run_codex_trial(
        ws, model=args.model, effort=args.effort, has_description=desc is not None,
        timeout=args.timeout, verbose=args.verbose,
    )

    print(f"\n[status] {status}")
    print(f"[time]   {meta.get('elapsed_sec')}s")
    print(f"[usage]  {meta.get('token_usage')}")
    if not rgx:
        return 1

    print(f"[regex]  {rgx}")

    # Score it the same way the eval does: recall on held-out TPs, FP rate on TNs.
    try:
        pat = re.compile(rgx)
    except re.error as e:
        print(f"[score]  INVALID REGEX: {e}")
        return 1

    seed_ok = bool(pat.search(seed))
    tp_hits = sum(1 for t in held_out if pat.search(t))
    fp_hits = sum(1 for t in tns if pat.search(t))
    print(f"[score]  matches seed: {seed_ok}")
    print(f"[score]  recall on held-out TPs: {tp_hits}/{len(held_out)}"
          + (f" = {tp_hits/len(held_out):.1%}" if held_out else " (n/a)"))
    print(f"[score]  false positives: {fp_hits}/{len(tns)}"
          + (f" = {fp_hits/len(tns):.2%}" if tns else " (n/a)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
