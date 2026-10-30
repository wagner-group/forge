"""Claude Code evaluation helpers."""
from .agent import (
    DEFAULT_CLAUDE_MODEL,
    check_claude_prerequisites,
    configure_claude_env,
    default_proxy_url,
    proxy_reachable,
    run_claude_code_trial,
)
from .prompts import (
    build_per_agent_prompt,
    default_dataset_root,
    load_per_agent_prompt_template,
    mapping_path,
    prompt_md_path,
    resolve_numeric_id,
    rule_name_from_file_path,
)
from .recovery import (
    collect_regex_candidates,
    is_max_turns_failure,
    last_regex_matching_seed,
    recover_last_matching_regex,
)
from .stream import StreamEventPrinter, extract_final_result
from .workspace import batch_result_path, prepare_trial_workspace

__all__ = [
    "DEFAULT_CLAUDE_MODEL",
    "StreamEventPrinter",
    "batch_result_path",
    "build_per_agent_prompt",
    "check_claude_prerequisites",
    "collect_regex_candidates",
    "configure_claude_env",
    "default_dataset_root",
    "default_proxy_url",
    "extract_final_result",
    "is_max_turns_failure",
    "last_regex_matching_seed",
    "load_per_agent_prompt_template",
    "mapping_path",
    "prepare_trial_workspace",
    "prompt_md_path",
    "proxy_reachable",
    "recover_last_matching_regex",
    "resolve_numeric_id",
    "rule_name_from_file_path",
    "run_claude_code_trial",
]
