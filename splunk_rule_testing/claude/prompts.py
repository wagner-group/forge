"""Load filename mapping and per-agent prompts for Claude Code evaluation."""
from __future__ import annotations

import os
import re
from typing import Dict, Optional


_MAPPING_LINE = re.compile(r"\|\s*`(\d+)\.json`\s*\|\s*`([\w]+)`\s*\|")


def load_filename_mapping(mapping_path: str) -> Dict[str, str]:
    """Return numeric id (e.g. '07') -> rule name."""
    mapping: Dict[str, str] = {}
    with open(mapping_path, encoding="utf-8") as f:
        for line in f:
            m = _MAPPING_LINE.search(line)
            if m:
                mapping[m.group(1)] = m.group(2)
    return mapping


def load_rule_to_numeric(mapping_path: str) -> Dict[str, str]:
    """Return rule name -> numeric id (e.g. '07')."""
    fwd = load_filename_mapping(mapping_path)
    return {rule: num for num, rule in fwd.items()}


def rule_name_from_file_path(file_path: str) -> str:
    base = os.path.basename(file_path)
    if base.endswith(".json"):
        return base[:-5]
    return base


def resolve_numeric_id(rule_name: str, mapping_path: str) -> Optional[str]:
    return load_rule_to_numeric(mapping_path).get(rule_name)


# The as-run prompt templates carry the absolute working directory of the machine
# the published runs were generated on. build_per_agent_prompt() rewrites that
# prefix to the caller's workspace root. The prefix is not recorded here, because
# it contains an account name: set $LEGACY_PROMPT_PREFIX to it when replaying one
# of those templates, or use the sed recipe in the README. Left unset, any
# absolute prefix already present in the template is rewritten generically.
DEFAULT_LEGACY_PROMPT_PREFIX = ""
GENERIC_LEGACY_PREFIX = r"/(?:home|data2)/[^/\s]+/[\w.-]+(?:/eval_\d+)?/"


def load_per_agent_prompt_template(prompt_md_path: str) -> str:
    """Extract the per-agent prompt body from AGENT_PROMPT_NO_DESCRIPTION_LOOP.md."""
    with open(prompt_md_path, encoding="utf-8") as f:
        content = f.read()

    marker = "## Per-agent prompt (substitute `{BATCH}`)"
    start = content.find(marker)
    if start == -1:
        raise ValueError(f"Per-agent prompt section not found in {prompt_md_path}")
    body = content[start + len(marker):].strip()
    # Drop trailing orchestrator-only sections if present
    for stop in ("## Orchestrator prompt", "---"):
        idx = body.find(stop)
        if idx != -1 and idx > 0:
            body = body[:idx].strip()
    return body.lstrip("\n")


def build_per_agent_prompt(
    template: str,
    *,
    workspace_root: str,
    batch_id: str,
) -> str:
    """Substitute paths and batch id; fix randomized folder name in template."""
    prompt = template.replace("{BATCH}", batch_id)
    prompt = prompt.replace("randomize_pre_filter/", "randomized/")
    # Replace example absolute paths with actual workspace root (inputs/CWD only).
    # The default is the absolute prefix carried by the as-run templates shipped
    # under results/claude/, so behaviour matches the published runs exactly when
    # $LEGACY_PROMPT_PREFIX is unset. Override it only if your templates carry a
    # different prefix.
    explicit = os.environ.get("LEGACY_PROMPT_PREFIX") or DEFAULT_LEGACY_PROMPT_PREFIX
    prompt = re.sub(
        re.escape(explicit) if explicit else GENERIC_LEGACY_PREFIX,
        workspace_root.rstrip("/") + "/",
        prompt,
    )
    prompt = re.sub(
        r"Work there as your CWD\.",
        f"Work in `{workspace_root}` as your CWD.",
        prompt,
        count=1,
    )
    # Ensure workspace path appears when template uses relative description
    if workspace_root not in prompt:
        prompt = prompt.replace(
            f"batch `{batch_id}` in ",
            f"batch `{batch_id}` in `{workspace_root}`. ",
            1,
        )
    return prompt


def default_dataset_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "log_rules_dataset_final"))


def mapping_path(dataset_root: str) -> str:
    return os.path.join(dataset_root, "md_files", "filename_mapping.md")


def prompt_md_path(dataset_root: str) -> str:
    return os.path.join(dataset_root, "md_files", "AGENT_PROMPT_NO_DESCRIPTION_LOOP.md")
