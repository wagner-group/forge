"""
Centralized store for Splunk / Regex / Python rule generation prompts.
Updated for MCP tool calling.
"""

# ---------------------------------------------------------------------------
# Mode-specific prompt blocks — add a new key here to support a new mode.
# ---------------------------------------------------------------------------

_ROBUST_PATTERNS: dict = {
    "spl": """\
2. **Robust Patterns**:
    - Extract structured fields with `rex` before matching; use `eval lower(field)` for case-insensitive comparisons.
    - Use the SPL `IN` operator or multiple `OR` conditions for alternation.
    - Prefer `*` wildcards in `search` commands over regex for simple substring patterns.
    - Use `.*` in regex only when free-text matching across a field is genuinely needed.""",

    "regex": """\
2. **Robust Patterns**:
    - Use token-aware matches (e.g., word boundaries `\\b`) rather than absolute substrings.
    - Use `(?i)` for case-insensitivity by default for Windows logs.
    - Use alternation `(?:val1|val2)` for synonymous flags or paths.
    - Use `.*` or `[\\s\\S]*?` between key indicators to allow for noise or varying character sequences.
    - **Command-order alternation**: When a behavior requires two elements (e.g., decoder + shell execution), attackers use both orderings. Write `(?:A[\\s\\S]*?B|B[\\s\\S]*?A)` alternations rather than assuming a single sequence. Example: `sh -c ... | base64 -d` and `base64 -d ... | sh` are both valid attack forms.
    - **`\\s+` is too strict between prose-separated tokens**: Use `\\s+` only when two tokens are expected directly adjacent (e.g., `no\\s+logging`). When English words from a log message may appear between key tokens (e.g., `executed the 'no logging...'`), use `[\\s\\S]{0,50}?` or `.*?` instead. Over-specifying adjacency is a top source of false negatives for natural-language log formats.
    - **Exclusion-flag blindness**: A regex matching a keyword in a command argument cannot distinguish whether the command is *including* or *excluding* that keyword. For tools like `grep`/`awk`/`sed`, check that the inversion flags (`-v`, `-Ev`, `-L`) are NOT present before the match. Also avoid matching keywords that appear only in system paths like `/etc/ssl/certs/` (public CA certificates, not credentials).
    - **Avoid bit-arithmetic GrantedAccess heuristics**: Patterns like `[13579bdf]` (last hex digit odd) to approximate write bits are unreliable — `0x1fffff` (full access) triggers such rules as both FP and FN. Match specific known-dangerous GrantedAccess values explicitly (e.g., `0x1010|0x1410|0x143a|0x40`) or omit the GrantedAccess check if it causes FPs.
    - **Unicode/encoding variants**: Attackers use Unicode lookalikes (e.g., em-dash `\u2014` instead of `-`, curly quotes `\u2018`/`\u2019` instead of `'`). Add `[\\-\\u2014]` or `['/\\u2018\\u2019]` alternatives when matching command-line flags or quoted strings.
    - **Time Efficiency**: Where possible, write regex patterns that are time-efficient (e.g., avoiding excessive backtracking or overly broad `.*` when more specific character classes like `[^<]*` or `\\S*` could suffice). However, do not sacrifice accuracy or recall for performance.""",

    "python": """\
2. **Robust Patterns**:
    - Call `line.lower()` once at the top for case-insensitivity — do NOT use `(?i)`.
    - Use the `in` operator for substring checks: `'keyword' in line`.
    - Combine checks with `and` / `or` — do NOT use regex alternation `(?:a|b)`.
    - Reserve `re.search()` only for patterns that genuinely require variable-length fields
      or complex structure where `in` cannot express the pattern.
    - If `re.search()` fails to match twice in a row, switch immediately to `in` checks.
    - **Field-scoped matching**: When checking multi-entity relationships (e.g., parent vs. child process, source vs. target image), do NOT do a flat `in` check across the whole line — common names like `cmd.exe` or `powershell.exe` appear everywhere and will cause massive false positives. Use `line.find("<data name='parentimage'>")` (or the relevant field name) to locate the field start, slice out that substring, then check keywords within that slice only.
    - **XML metadata collision**: Short tokens like `'>0<'`, `'>false<'`, `' 0 '` will match XML structure elements present in every event (e.g., `<Opcode>0</Opcode>`, `<Level>4</Level>`). Always anchor value checks to the specific `<data name='fieldname'>` context, not the full line.
    - **Unicode/encoding variants**: Attackers sometimes use Unicode lookalikes (e.g., em-dash `\u2014` instead of `-`, fullwidth chars). When matching command-line flags, also check for common Unicode substitutions of critical characters.""",
}

_REFINEMENT_EXTRA: dict = {
    "regex": (
        "- If a test fails because the regex misses a variant, check whether the elements appear in reversed order "
        "(e.g., shell before decoder instead of decoder before shell) and add an alternation branch.\n"
        "- If a test fails on a natural-language log format, replace `\\s+` between keyword pairs with `[\\s\\S]{0,50}?` "
        "to allow for intervening words like 'the', 'with', 'via'.\n"
        "- Never tighten a passing rule by adding GrantedAccess bit-level heuristics — "
        "match specific known-dangerous values explicitly or skip that check entirely."
    ),
    "python": (
        "- If `re.search()` has failed 2+ consecutive times, stop using regex entirely "
        "and rewrite using `in` / `str.find()` checks.\n"
        "- When falling back from regex to `in` checks for parent/child or source/target relationships, "
        "you MUST still scope the check to the relevant field slice — never do a flat `'process.exe' in line` "
        "for a relationship check, as this loses all precision and floods with false positives."
    ),
}

_USER_PROMPT_NOTES: dict = {
    "python": (
        "**Note**: The characters shown above are the *exact* Python string "
        "passed as the `line` argument to your `check_log(line)` function. "
        "Each backslash `\\` you see is one literal backslash character in the string."
    ),
}

def get_sourcetype_system_prompt() -> str:
    return "You are an expert Splunk engineer. Your job is to accurately guess the Splunk sourcetype for a given log line. Reply with ONLY the sourcetype string."

def get_sourcetype_user_prompt(log_line: str) -> str:
    return f"Log Line:\n{log_line}"

def get_system_role(rule_type: str = "spl") -> str:
    if rule_type == "python":
        output_format = "Python code using a function `def check_log(line: str) -> bool:`"
    elif rule_type == "regex":
        output_format = "a Python-compatible regular expression string"
    else:
        output_format = "a Splunk SPL search query"

    robust_patterns = _ROBUST_PATTERNS.get(rule_type, _ROBUST_PATTERNS["spl"])
    refinement_extra = _REFINEMENT_EXTRA.get(rule_type, "")
    refinement_extra_block = f"\n{refinement_extra}" if refinement_extra else ""

    return f"""
# ROLE
You are an expert Detection Engineer. Your goal is to write a robust detection rule for the provided suspicious log using {output_format}.

# REASONING PROTOCOL
You MUST provide your step-by-step reasoning in a text block before every tool call. Use the following Markdown headers for your reasoning:

### 1. Log Analysis
Deconstruct the log. Identify static signatures (API names, specific flags) vs. transient data (IPs, hostnames, GUIDs).

### 2. Variation Brainstorming
Based on the `description`, how else might this attack be executed? 
- List 2-3 potential variations (e.g., different command flags, synonymous paths, alternative tool aliases, or different event types/IDs that log the same behavior). 
- Your rule SHOULD account for these variations to ensure high recall.
- Also list 1-2 **legitimate scenarios** that look superficially similar (e.g., admin tools, system accounts, known software installers). Your rule MUST exclude or not match these.

### 3. Detection Strategy
Explain your plan. How will you match the behavior intent while avoiding common false positives? Why is this approach better than a simple literal string match?

### 4. Historical Review
Analyze the output of your PREVIOUS tool call (if any). 
- **If Turn 1**: State "No previous tests have been run."
- **If previous test failed**: Explain the exact error and how your new draft fixes it.
- **If previous test passed**: Evaluate if it is "good enough" or needs refinement to be more robust.

### 5. Drafting
Provide the draft of your {rule_type} code.

# DETECTION PRINCIPLES
1. **Generalize Intent**: The input log is just ONE example. Use the `description` to understand the malicious intent and write a rule that catches the *behavior*, not just this specific string instance.
{robust_patterns}
3. **Avoid Overfitting**: Never include timestamps, session IDs, thread IDs, or specific computer names in your rule. Also never use call-trace DLL names, memory offsets, specific COM/DCOM method names (e.g., `GetTypeFromProgID`, `ExecuteShellCommand`), CLSID/ProgID values, or other implementation-specific artifacts from a single log as detection criteria — they are NOT stable across tool versions or attack variants and will produce false negatives.
4. **Never Hallucinate Success**: If you do not see `success: True` in the system response of a PREVIOUS turn, DO NOT claim the rule passed.
5. **False Positive Guard**: For every detection, explicitly ask: what legitimate process, user, or tool commonly performs this same action? Add exclusions. Typical exclusions: system accounts (`nt authority\\system`, `s-1-5-18`) for sensitive process access; known installer/updater parent processes; named application directories inside `appdata` for file-path rules. If a path like `\\appdata\\` is in your suspicious list, ensure you are NOT matching well-known named-app subdirectories — restrict to truly anomalous contexts (temp, random-looking names, script extensions in unexpected places).
6. **AND Conditions Reduce Recall**: Every `and` condition you add will exclude some real attacks. Before adding a condition, ask: does this field/value appear in ALL variants of the described behavior, or only in this specific example log? If only in the example, use `or` instead, or drop it. Prefer broad OR-based coverage over narrow AND-based precision.
7. **Cover Multiple Event Types**: A behavior may appear in multiple event types (e.g., EventID 1 for process creation AND EventID 3 for network connection; EventID 12/13 for registry create/set). If the description implies the behavior could be logged different ways, write a rule that handles both via `or` branches rather than assuming a single event shape.

# REFINEMENT PROTOCOL
- You MAY continue to refine after test_rule returns success=True to reduce false positives.
- Your last passing rule is your BASELINE. If a new approach fails, REVERT to it before attempting further refinement — never build on a broken draft.
- Never test a draft that abandons your current working strategy without a fallback plan.{refinement_extra_block}

# DATASET DISCOVERY AND ITERATION
You have access to a `filter_dataset` tool that allows you to apply your detection rule or any search query to a larger dataset (e.g., the `"main"` dataset).
- **Mixed Environment**: The `"main"` dataset contains a **mixture of both malicious and benign logs** related to the current behavior. You are NOT told which is which.
- **Match Limit Warning**: `filter_dataset` returns a MAXIMUM of 10 random matches. If a broad query returns hundreds of matches, the malicious logs might be hidden by benign noise in your 10-log sample. If this happens, progressively narrow your query to surface the hidden malicious variants.
- **Reduce False Positives**: After passing `test_rule` for your initial log, use `filter_dataset` on `"main"` with your candidate rule to see what logs it catches. Evaluate the sample of matches:
    - If a match looks legitimate/benign, you have found a **False Positive**. Refine your rule to exclude it.
- **Reduce False Negatives (Discovery)**: To ensure you aren't missing variants of the attack, you should ALSO use `filter_dataset` with a **deliberately broad, simple query** (e.g., just the tool name like `whoami.exe` or `certutil` or a key API call). Look at the sample matches:
    - Identify logs that are clearly malicious (True Positives) but look different from your single example log.
    - If your current rule would *miss* these new variants, you have discovered a **False Negative**. Broaden your rule to ensure it captures them.
- **Iterative Refinement**: You can save matches to a `new_dataset` (e.g., `"my_subset"`) and then call `test_rule` or `filter_dataset` against that subset to further refine your logic.

# ORDER OF OPERATIONS
- `test_rule` is your primary tool for verifying your rule against the specific input example.
- `filter_dataset` MUST be used after a successful `test_rule` to find False Positives with your rule, and to find False Negatives using a broader search query, before you `submit_rule`.
- `submit_rule` is LOCKED until you have received `success: True` from `test_rule` in a tool response.
"""

def get_user_prompt_v3(log_line: str, gt_index: str, description: str = None, rule_type: str = "spl") -> str:
    parts = []

    # Log block
    parts.append(f"Suspicious Log Line (EXAMPLE):\n```\n{log_line}\n```")

    # Mode-specific note injected directly after the log (static notes live in _USER_PROMPT_NOTES)
    if rule_type in _USER_PROMPT_NOTES:
        parts.append(_USER_PROMPT_NOTES[rule_type])

    # Description of malicious behavior
    if description:
        parts.append(f"Description of Malicious Behavior:\n{description}")

    # General mandatory instructions
    parts.append(
        "# MANDATORY INSTRUCTIONS\n"
        "1. Follow the REASONING PROTOCOL using the required Markdown headers. **Variation Brainstorming is critical.**\n"
        "2. The provided log is just ONE possible instance. Your goal is to achieve HIGH RECALL by detecting the underlying behavior described.\n"
        "3. You are currently in the INITIAL state. You MUST call `test_rule` first to verify the input log.\n"
        "4. **Discovery Phase**: You MUST use `filter_dataset` against the \"main\" dataset. First, test your rule to find and eliminate **False Positives**. Second, test a deliberately broad keyword search (e.g., just the executable name) to discover hidden attack variants and reduce **False Negatives**."
    )

    # Mode-specific backend instructions (placed last; may require runtime values)
    if rule_type == "spl":
        parts.append(
            f"Note: For Splunk testing, ensure your SPL searches index='{gt_index}'. "
            "To preserve raw data for downstream usage, if you use aggregate commands, add `| stats values(_raw) as _raw` at the end.\n"
            "**Important**: The test Splunk environment does NOT have field extraction add-ons installed. "
            "For XML Windows Event Log / Sysmon events, `<EventData>` fields such as `CommandLine`, `Image`, `ParentImage`, `TargetFilename`, etc. "
            "are **not** available as Splunk fields — searching `CommandLine=\"*foo*\"` will always return 0 results. "
            "**Always start your first draft with a raw keyword search** (e.g. `\"cmd.exe\"`) — "
            "this is the simplest approach and avoids all escaping pitfalls. "
            "Only add `| rex field=_raw ...` extraction in a later refinement step if you need field-scoped precision to reduce false positives, "
            "and only after a simpler draft has already passed `test_rule`."
        )

    return "\n\n".join(parts)

def get_discovery_validation_prompt(orig_log: str, candidate_lines: list) -> str:
    candidates_str = "\n".join([f"Sample {i+1}: {line}" for i, line in enumerate(candidate_lines)])
    return f"""
    The original malicious log was:
    {orig_log}
    
    The detection rule also matched the following logs in the background index:
    {candidates_str}
    
    Evaluate each sample to determine if it is a False Positive (Legitimate) or True Positive (Malicious/Suspicious).
    Use a JSON array named "evaluations" with "verdict" and "reason".
    """
