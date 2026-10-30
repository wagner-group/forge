"""Invoke Claude Code CLI for regex rule generation trials."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from .prompts import build_per_agent_prompt, load_per_agent_prompt_template
from .stream import StreamEventPrinter, extract_final_result
from .workspace import batch_result_path
from detection import Colors

DEFAULT_PROXY_PORT = 4000
# Placeholder key for a LOCAL LiteLLM proxy. Not a credential: the local proxy
# accepts any value. Deliberately not "sk-"-prefixed so it is not mistaken for
# an Anthropic key by secret scanners or readers.
DEFAULT_PROXY_KEY = "local-litellm-proxy-key"
# Fallback only. The published baseline runs passed the model explicitly on the
# command line: Claude Opus 4.7 at maximum reasoning effort, as reported in the
# paper. This default was never used for any published result.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"


def default_proxy_url() -> str:
    port = os.environ.get("CLAUDE_GEMINI_PROXY_PORT", str(DEFAULT_PROXY_PORT))
    return f"http://127.0.0.1:{port}"


def proxy_reachable(base_url: str, timeout_sec: float = 2.0) -> bool:
    """Return True if a LiteLLM (or compatible) proxy responds at /health."""
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_sec) as resp:
            return resp.status in (200, 401)
    except urllib.error.HTTPError as e:
        return e.code in (200, 401)
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _has_claude_auth(env: Mapping[str, str]) -> bool:
    """True when Claude Code can authenticate (direct API key or LiteLLM gateway)."""
    if env.get("ANTHROPIC_API_KEY"):
        return True
    if env.get("ANTHROPIC_AUTH_TOKEN") and env.get("ANTHROPIC_BASE_URL"):
        return True
    if env.get("CLAUDE_CODE_USE_VERTEX") == "1":
        return True
    return False


def configure_claude_env(
    env: Optional[MutableMapping[str, str]] = None,
    *,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    use_gemini_proxy: bool = True,
    verbose: bool = False,
) -> Dict[str, str]:
    """Return env dict with Claude Code auth configured.

    When no Anthropic credentials are set and use_gemini_proxy is True, auto-detect
    a local LiteLLM proxy at the default URL (CLAUDE_GEMINI_PROXY_URL /
    CLAUDE_GEMINI_PROXY_PORT; see env.example). The proxy itself is a local
    service and is not part of this artifact.
    """
    merged: Dict[str, str] = dict(env if env is not None else os.environ)

    if _has_claude_auth(merged):
        if not merged.get("ANTHROPIC_API_KEY") and merged.get("ANTHROPIC_AUTH_TOKEN"):
            merged["ANTHROPIC_API_KEY"] = merged["ANTHROPIC_AUTH_TOKEN"]
        return merged

    if not use_gemini_proxy:
        return merged

    url = proxy_url or merged.get("CLAUDE_GEMINI_PROXY_URL") or default_proxy_url()
    key = (
        proxy_key
        or merged.get("LITELLM_MASTER_KEY")
        or merged.get("ANTHROPIC_AUTH_TOKEN")
        or DEFAULT_PROXY_KEY
    )

    if proxy_reachable(url):
        merged["ANTHROPIC_BASE_URL"] = url
        merged["ANTHROPIC_AUTH_TOKEN"] = key
        merged["ANTHROPIC_API_KEY"] = key
        if verbose:
            print(
                f"{Colors.OKBLUE}[INFO] Using Gemini LiteLLM proxy at {url} "
                f"(model should be a Claude alias, e.g. {DEFAULT_CLAUDE_MODEL}){Colors.ENDC}"
            )
    return merged


def check_claude_prerequisites(
    claude_bin: str = "claude",
    *,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    use_gemini_proxy: bool = True,
) -> Optional[str]:
    """Return error message if Claude Code cannot run, else None."""
    import shutil

    if not shutil.which(claude_bin):
        return (
            f"Claude Code CLI '{claude_bin}' not found on PATH.\n"
            "Install from https://code.claude.com/docs/en/setup and ensure `claude` is available."
        )

    env = configure_claude_env(
        proxy_url=proxy_url,
        proxy_key=proxy_key,
        use_gemini_proxy=use_gemini_proxy,
        verbose=True,
    )
    if _has_claude_auth(env):
        return None

    url = proxy_url or default_proxy_url()
    if use_gemini_proxy and not proxy_reachable(url):
        return (
            f"Gemini LiteLLM proxy is not reachable at {url}.\n"
            "Either set ANTHROPIC_API_KEY to run Claude Code against Anthropic\n"
            "directly, or start a LiteLLM proxy that serves Gemini behind Claude\n"
            "model aliases at that URL and re-run (auto-detect configures the env\n"
            "once the proxy is up). The proxy is a local service; no launcher for\n"
            "it ships with this artifact."
        )

    return (
        "Claude Code auth is not configured.\n"
        "Set ANTHROPIC_API_KEY, or start a local LiteLLM proxy and set "
        "ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN "
        "(see CLAUDE_GEMINI_PROXY_PORT in env.example)."
    )


def _build_claude_cmd(
    claude_bin: str,
    prompt: str,
    claude_model: str,
    *,
    verbose: bool,
) -> List[str]:
    cmd = [
        claude_bin,
        "--bare",
        "-p",
        prompt,
        "--model",
        claude_model,
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Read,Write,Bash",
        "--no-session-persistence",
    ]
    if verbose:
        cmd.extend([
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--settings",
            '{"showThinkingSummaries": true}',
        ])
    else:
        cmd.extend(["--output-format", "json"])
    return cmd


def _write_claude_log(log_path: Optional[str], combined_log: str) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(combined_log)


def _run_claude_buffered(
    cmd: List[str],
    *,
    workspace_root: str,
    env: Dict[str, str],
    timeout_sec: int,
) -> Tuple[str, int]:
    proc = subprocess.run(
        cmd,
        cwd=workspace_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    combined_log = ""
    if proc.stdout:
        combined_log += proc.stdout
    if proc.stderr:
        combined_log += ("\n" if combined_log else "") + proc.stderr
    return combined_log, proc.returncode


def _run_claude_streaming(
    cmd: List[str],
    *,
    workspace_root: str,
    env: Dict[str, str],
    timeout_sec: int,
) -> Tuple[str, int]:
    proc = subprocess.Popen(
        cmd,
        cwd=workspace_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    printer = StreamEventPrinter()
    log_lines: List[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_lines.append(line.rstrip("\n"))
            printer.process_line(line)
        returncode = proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    return "\n".join(log_lines), returncode


def _failure_tail(combined_log: str) -> str:
    final = extract_final_result(combined_log)
    if final is not None:
        return json.dumps(final, ensure_ascii=False)[-500:]
    return combined_log.strip()[-500:]


def _finalize_trial(
    combined_log: str,
    returncode: int,
    workspace_root: str,
    batch_id: str,
    numeric_id: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if returncode != 0:
        return None, f"Claude Code exited {returncode}: {_failure_tail(combined_log)}"

    result_path = batch_result_path(workspace_root, batch_id)
    if not os.path.isfile(result_path):
        return None, f"Claude did not write expected output: {result_path}"

    try:
        with open(result_path, encoding="utf-8") as f:
            batch_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, f"Could not read Claude result JSON: {e}"

    entry = _find_result_entry(batch_data, numeric_id)
    if entry is None:
        return None, f"No result entry for numeric id {numeric_id} in {result_path}"
    if not entry.get("regex"):
        err = entry.get("error", "regex is null")
        return entry, f"Generation Failed: {err}"

    return entry, "Success"


def run_claude_code_trial(
    workspace_root: str,
    batch_id: str,
    numeric_id: str,
    *,
    prompt_template: str,
    claude_bin: str = "claude",
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    timeout_sec: int = 1800,
    verbose: bool = False,
    log_path: Optional[str] = None,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    use_gemini_proxy: bool = True,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Spawn Claude Code and return (result entry dict, status message)."""
    prompt = build_per_agent_prompt(
        prompt_template,
        workspace_root=workspace_root,
        batch_id=batch_id,
    )

    cmd = _build_claude_cmd(
        claude_bin,
        prompt,
        claude_model,
        verbose=verbose,
    )

    if verbose:
        print(f"{Colors.OKCYAN}[DEBUG] Claude cmd: {' '.join(cmd[:6])} ... (stream-json){Colors.ENDC}")

    env = configure_claude_env(
        proxy_url=proxy_url,
        proxy_key=proxy_key,
        use_gemini_proxy=use_gemini_proxy,
        verbose=verbose,
    )
    try:
        if verbose:
            combined_log, returncode = _run_claude_streaming(
                cmd,
                workspace_root=workspace_root,
                env=env,
                timeout_sec=timeout_sec,
            )
        else:
            combined_log, returncode = _run_claude_buffered(
                cmd,
                workspace_root=workspace_root,
                env=env,
                timeout_sec=timeout_sec,
            )
    except subprocess.TimeoutExpired:
        return None, f"Claude Code timed out after {timeout_sec}s"
    except OSError as e:
        return None, f"Failed to spawn Claude Code: {e}"

    _write_claude_log(log_path, combined_log)
    return _finalize_trial(
        combined_log,
        returncode,
        workspace_root,
        batch_id,
        numeric_id,
    )


def _find_result_entry(batch_data: Dict[str, Any], numeric_id: str) -> Optional[Dict[str, Any]]:
    results = batch_data.get("results") or []
    candidates = {numeric_id, numeric_id.lstrip("0") or "0", numeric_id.zfill(2)}
    for item in results:
        name = str(item.get("name", ""))
        if name in candidates or name.zfill(2) in candidates:
            return item
    return None
