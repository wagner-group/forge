"""Recover regex rules from Claude Code workspaces after max_turns failures."""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

from .workspace import batch_result_path

_RE_JSON_REGEX = re.compile(r'"regex"\s*:\s*"((?:\\.|[^"\\])*)"')
_RE_RE_SEARCH = re.compile(
    r"re\.search\s*\(\s*r?(?P<q>['\"])(?P<p>(?:\\.|(?!\1).)*)(?P=q)",
    re.DOTALL,
)
_RE_RE_COMPILE = re.compile(
    r"re\.compile\s*\(\s*r?(?P<q>['\"])(?P<p>(?:\\.|(?!\1).)*)(?P=q)",
    re.DOTALL,
)
_RE_REGEX_ASSIGN = re.compile(
    r"(?:^|\s)regex\s*=\s*r?(?P<q>['\"])(?P<p>(?:\\.|(?!\1).)*)(?P=q)",
    re.MULTILINE | re.DOTALL,
)
_RE_FENCED_CODE = re.compile(r"```(?:regex|python|)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def is_max_turns_failure(status: str, log_path: str) -> bool:
    """Return True if the Claude Code run stopped due to max_turns."""
    status_lower = (status or "").lower()
    if "max_turns" in status_lower or "error_max_turns" in status_lower:
        return True
    if "reached maximum number of turns" in status_lower:
        return True

    if not log_path or not os.path.isfile(log_path):
        return False

    try:
        with open(log_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return False

    if "max_turns" in content or "error_max_turns" in content:
        return True
    if "Reached maximum number of turns" in content:
        return True

    try:
        payload = json.loads(content.strip())
    except json.JSONDecodeError:
        return False

    if payload.get("terminal_reason") == "max_turns":
        return True
    subtype = payload.get("subtype") or ""
    if isinstance(subtype, str) and "max_turns" in subtype:
        return True
    errors = payload.get("errors") or []
    return any("maximum number of turns" in str(err).lower() for err in errors)


def _decode_python_string(raw: str) -> str:
    try:
        return bytes(raw, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return raw


def _decode_json_string(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return _decode_python_string(raw)


def _append_unique(candidates: List[Tuple[str, str]], pattern: str, source: str) -> None:
    if not pattern or not pattern.strip():
        return
    if candidates and candidates[-1][0] == pattern:
        return
    candidates.append((pattern, source))


def _walk_json_for_regex(obj: object, candidates: List[Tuple[str, str]], source: str) -> None:
    if isinstance(obj, dict):
        regex = obj.get("regex")
        if isinstance(regex, str) and regex.strip():
            _append_unique(candidates, regex, source)
        for value in obj.values():
            _walk_json_for_regex(value, candidates, source)
    elif isinstance(obj, list):
        for item in obj:
            _walk_json_for_regex(item, candidates, source)


def _extract_from_text(text: str, source: str) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    for match in _RE_JSON_REGEX.finditer(text):
        _append_unique(found, _decode_json_string(match.group(1)), f"{source}:json_regex")
    for match in _RE_RE_SEARCH.finditer(text):
        _append_unique(found, _decode_python_string(match.group("p")), f"{source}:re.search")
    for match in _RE_RE_COMPILE.finditer(text):
        _append_unique(found, _decode_python_string(match.group("p")), f"{source}:re.compile")
    for match in _RE_REGEX_ASSIGN.finditer(text):
        _append_unique(found, _decode_python_string(match.group("p")), f"{source}:regex_assign")
    for match in _RE_FENCED_CODE.finditer(text):
        block = match.group(1)
        for item in _extract_from_text(block, f"{source}:code_block"):
            _append_unique(found, item[0], item[1])
    return found


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def collect_regex_candidates(
    workspace_root: str,
    batch_id: str,
    claude_log_path: str,
) -> List[Tuple[str, str]]:
    """Collect regex candidates in document order with their source labels."""
    candidates: List[Tuple[str, str]] = []

    log_text = _read_text(claude_log_path)
    if log_text:
        try:
            payload = json.loads(log_text.strip())
            _walk_json_for_regex(payload, candidates, "claude_run.log:json")
            result_text = payload.get("result")
            if isinstance(result_text, str):
                for pattern, source in _extract_from_text(result_text, "claude_run.log:result"):
                    _append_unique(candidates, pattern, source)
        except json.JSONDecodeError:
            pass
        for pattern, source in _extract_from_text(log_text, "claude_run.log"):
            _append_unique(candidates, pattern, source)

    result_path = batch_result_path(workspace_root, batch_id)
    if os.path.isfile(result_path):
        try:
            with open(result_path, encoding="utf-8") as f:
                batch_data = json.load(f)
            _walk_json_for_regex(batch_data, candidates, os.path.basename(result_path))
        except (json.JSONDecodeError, OSError):
            pass

    py_files: List[str] = []
    for root, _dirs, files in os.walk(workspace_root):
        for name in files:
            if name.endswith(".py"):
                py_files.append(os.path.join(root, name))
    py_files.sort(key=lambda p: (os.path.getmtime(p), p))

    for py_path in py_files:
        rel = os.path.relpath(py_path, workspace_root)
        for pattern, _source in _extract_from_text(_read_text(py_path), rel):
            kind = _source.split(":", 1)[-1] if ":" in _source else _source
            _append_unique(candidates, pattern, f"{rel}:{kind}")

    return candidates


def last_regex_matching_seed(
    candidates: List[Tuple[str, str]],
    seed_tp: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Return the last regex candidate that matches the seed true positive."""
    last_match: Optional[str] = None
    last_source: Optional[str] = None
    for pattern, source in candidates:
        try:
            if re.search(pattern, seed_tp, re.DOTALL):
                last_match = pattern
                last_source = source
        except re.error:
            continue
    return last_match, last_source


def recover_last_matching_regex(
    workspace_root: str,
    batch_id: str,
    seed_tp: str,
    claude_log_path: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Recover the last seed-matching regex from a max_turns workspace."""
    candidates = collect_regex_candidates(workspace_root, batch_id, claude_log_path)
    return last_regex_matching_seed(candidates, seed_tp)
