import argparse
import json
import re
import sys
import time
import os
from typing import List, Dict, Tuple, Any, Optional
from dotenv import load_dotenv
import prompts

# Re-export detection symbols so callers using `import agent as harness`
# can still call SplunkEngine, RegexEngine, etc.
from detection import *  # noqa: F401,F403
from detection import Colors
from detection.regex import RegexEngine
from detection.python import PythonEngine
from tool_registry import ToolRegistry

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL_NAME = "gemini-3.1-pro-preview"

# Provider switch. "gemini" keeps the original behaviour; "openai" routes through
# llm_openai (GPT-5.6 family via the Responses API). Set via --provider or the
# FORGE_PROVIDER env var.
PROVIDER = os.environ.get("FORGE_PROVIDER", "gemini")
OPENAI_MODEL = os.environ.get("FORGE_OPENAI_MODEL", "gpt-5.6-luna")
OPENAI_EFFORT = os.environ.get("FORGE_OPENAI_EFFORT", "none")

_api_call_count = 0


def reset_api_call_count() -> None:
    """Reset the successful LLM API call counter (used per evaluation trial)."""
    global _api_call_count
    _api_call_count = 0
    if PROVIDER == "openai":
        import llm_openai
        llm_openai.USAGE.reset()


def get_api_call_count() -> int:
    """Return successful generate_content calls since the last reset (retries excluded)."""
    return _api_call_count


def _bump_call_count() -> None:
    global _api_call_count
    _api_call_count += 1


def get_token_usage() -> dict:
    """Return token usage since the last reset. Empty dict on providers that don't report it."""
    if PROVIDER == "openai":
        import llm_openai
        return llm_openai.USAGE.as_dict()
    return {}


def _make_tool_result(function_call, name: str, result: dict):
    """Build the provider-specific tool-result payload for one executed call."""
    if PROVIDER == "openai":
        import llm_openai
        return llm_openai.make_tool_result(function_call, result)
    from google.genai import types
    return types.Part.from_function_response(name=name, response=result)


# -----------------------------------------------------------------------------
# Gemini Interaction
# -----------------------------------------------------------------------------

def infer_sourcetype(log_line: str, api_key: str, vertexai: bool = False, vertex_project: str = os.environ.get("VERTEX_PROJECT", "")) -> str:
    """Infers the Splunk sourcetype for a given log line using Gemini."""
    system_prompt = prompts.get_sourcetype_system_prompt()
    user_prompt = prompts.get_sourcetype_user_prompt(log_line)
 
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
 
    inferred = call_llm(messages, api_key, vertexai=vertexai, vertex_project=vertex_project).text.strip()
 
    if inferred.startswith("```") and inferred.endswith("```"):
        inferred = inferred.split("\\n")[1].strip()
 
    return inferred
 
def call_llm(messages: List[Dict[str, Any]], api_key: str, response_format: Optional[Dict[str, str]] = None, max_retries: int = 6, vertexai: bool = False, vertex_project: str = os.environ.get("VERTEX_PROJECT", ""), tools: Optional[List[Any]] = None, allowed_tools: Optional[List[str]] = None) -> Any:
    if PROVIDER == "openai":
        import llm_openai
        resp = llm_openai.call_llm(
            messages,
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=OPENAI_MODEL,
            effort=OPENAI_EFFORT,
            tools=llm_openai.build_tool_schemas() if tools else None,
            allowed_tools=allowed_tools,
            max_retries=max_retries,
        )
        _bump_call_count()
        return resp

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print(f"{Colors.FAIL}Error: 'google-genai' library not installed.{Colors.ENDC}")
        sys.exit(1)
 
    if vertexai:
        from google.genai.types import HttpOptions
        client = genai.Client(
            vertexai=True,
            project=vertex_project,
            location='global',
            http_options=HttpOptions(api_version='v1')
        )
    else:
        client = genai.Client(api_key=api_key, http_options={'timeout': 180000})

    system_instruction = ""
    contents = []

    for m in messages:
        if m['role'] == 'system':
            system_instruction += m['content'] + "\n"
        elif m['role'] == 'user':
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m['content'])]))
        elif m['role'] == 'assistant':
            if 'model_content' in m:
                # Replay the exact Content object returned by the model (preserves all Parts)
                contents.append(m['model_content'])
            elif 'function_call' in m:
                # Legacy fallback: wrap a bare FunctionCall object into a Part
                fc = m['function_call']
                contents.append(types.Content(role="model", parts=[types.Part(function_call=fc)]))
            else:
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=m['content'])]))
        elif m['role'] == 'function':
            contents.append(types.Content(role="user", parts=[m['function_response']]))
        elif m['role'] == 'function_responses':
            # Bundled responses for a multi-call turn: must be ONE Content with all parts.
            contents.append(types.Content(role="user", parts=m['parts']))

    config_kwargs = {
        "system_instruction": system_instruction,
        "temperature": 0.2,
        "max_output_tokens": 16384
    }
    if response_format and response_format.get("type") == "json_object":
        config_kwargs["response_mime_type"] = "application/json"
    if tools:
        # We pass the full toolset to avoid internal SDK crashes when processing history.
        config_kwargs["tools"] = tools
        
        # We force ANY mode to ensure the model actually calls a tool rather than just talking.
        # We restrict which tool can be called via allowed_function_names.
        tool_config_kwargs = {"mode": "ANY"}
        if allowed_tools:
            tool_config_kwargs["allowed_function_names"] = allowed_tools
            
        config_kwargs["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(**tool_config_kwargs)
        )
        # Disable automatic function calling to allow our manual agent loop to manage turns
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            global _api_call_count
            _api_call_count += 1
            return response
        except Exception as e:
            last_error = e
            is_retryable = any(code in str(e) for code in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "Quota", "Server disconnected"))
            
            if not is_retryable or attempt == max_retries:
                if not is_retryable:
                    import traceback
                    print(f"{Colors.FAIL}[ERROR] Non-retryable error in call_llm: {e}{Colors.ENDC}")
                    traceback.print_exc()
                break
            
            wait = 2 ** attempt * 5
            print(f"{Colors.WARNING}[WARN] Gemini API error (attempt {attempt+1}/{max_retries+1}): {e}. Retrying in {wait}s...{Colors.ENDC}")
            time.sleep(wait)

    raise Exception(f"LLM Call failed: {last_error}")

# -----------------------------------------------------------------------------
# History Logging
# -----------------------------------------------------------------------------

def record_query_history(log_line: str, final_rule: Optional[str], match_count: int, status: str, sample_matches: Optional[List[Dict[str, Any]]] = None):
    """Records the session details to an individual timestamped JSON file in the history/ directory."""
    history_dir = "history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    history_file = os.path.join(history_dir, f"query_{timestamp_str}.json")

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_log": log_line,
        "final_spl": final_rule,
        "matches": match_count,
        "status": status,
        "sample_matches": sample_matches or []
    }

    with open(history_file, 'w') as f:
        json.dump(entry, f, indent=2)
    print(f"{Colors.OKGREEN}[INFO] Session details recorded to {history_file}{Colors.ENDC}")

# -----------------------------------------------------------------------------
# Main Harness Logic
# -----------------------------------------------------------------------------

# Import Splunk functions into local namespace for use within this module.
# (The wildcard import above makes them available, but explicit references
#  via the module help readability in this file.)
from detection import (
    SplunkEngine, RegexEngine, PythonEngine, Colors
)


def generate_rule_for_log(
    log_line: str,
    gt_index: str,
    api_key: str,
    splunk_engine: Optional[SplunkEngine] = None,
    verbose_llm: bool = False,
    description: Optional[str] = None,
    regex_engine: Optional[RegexEngine] = None,
    python_engine: Optional[PythonEngine] = None,
    vertexai: bool = False,
    vertex_project: str = os.environ.get("VERTEX_PROJECT", ""),
    main_dataset: str = "",
    no_description: bool = False,
) -> Tuple[Optional[str], str]:
    """Generate and validate a detection rule for a given log line using an Agentic MCP loop.

    Returns:
        Tuple[Optional[str], str]: (validated_rule, status_message)
    """
    if PROVIDER != "openai":
        from google.genai import types

    if python_engine:
        rule_type = "python"
        rule_label = "Python script"
        active_engine = python_engine
    elif regex_engine:
        rule_type = "regex"
        rule_label = "regex"
        active_engine = regex_engine
    else:
        rule_type = "spl"
        rule_label = "SPL"
        active_engine = splunk_engine

    registry = ToolRegistry(engine=active_engine, target_log=log_line, gt_index=gt_index, main_dataset=main_dataset)

    system_prompt = prompts.get_system_role(rule_type=rule_type)
    
    # If no_description is set, we suppress the human-written explanation
    effective_description = description if not no_description else None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompts.get_user_prompt_v3(
            log_line, gt_index, effective_description, rule_type=rule_type,
        )},
    ]

    print(f"\n{Colors.OKBLUE}[STEP 2] Generating and validating {rule_label}...{Colors.ENDC}")

    max_agent_steps = 25
    for step in range(max_agent_steps):
        print(f"Agent Loop Step {step+1}/{max_agent_steps}...")
        
        # Inject turn-based guidance to prevent hallucinations
        if step == 0:
            history_nudge = {
                "role": "user", 
                "content": "REMINDER: You are at Step 1. No tests have been run yet. Follow the 'Log Analysis' and 'Detection Strategy' headers. Do NOT claim the rule has passed."
            }
            messages.append(history_nudge)

        # Inject step-budget warning near the end of the budget
        if step == max_agent_steps - 3 and not registry.is_done:
            steps_left = max_agent_steps - step
            budget_nudge = {
                "role": "user",
                "content": (
                    f"SYSTEM ALERT: You have {steps_left} steps remaining. "
                    "If you have a passing rule (success=True from any previous test_rule call), "
                    "you MUST call submit_rule() NOW to lock in your best rule before the budget expires."
                )
            }
            messages.append(budget_nudge)

        try:
            allowed = registry.get_allowed_tool_names()
            if verbose_llm:
                print(f"{Colors.OKCYAN}[DEBUG] Allowed tools for this turn: {allowed}{Colors.ENDC}")
                
            current_tools = registry.get_allowed_tool_names()
            resp = call_llm(
                messages, api_key,
                vertexai=vertexai,
                vertex_project=vertex_project,
                tools=registry.get_tool_list(),
                allowed_tools=current_tools
            )
        except Exception as e:
            print(f"{Colors.WARNING}[WARN] LLM Call failed: {e}{Colors.ENDC}")
            return None, f"LLM Call Failed: {e}"
        
        # Defensive check for blocked or empty responses
        if not resp.candidates or not resp.candidates[0].content:
            finish_reason = "UNKNOWN"
            if resp.candidates and resp.candidates[0].finish_reason:
                finish_reason = resp.candidates[0].finish_reason
            
            error_msg = f"Gemini returned no candidates or content (Finish Reason: {finish_reason})."
            if finish_reason == "SAFETY":
                error_msg += " The response was likely blocked by safety filters."
            
            print(f"{Colors.WARNING}[WARN] {error_msg}{Colors.ENDC}")
            # If we were in ANY mode and it was blocked, we can't proceed with this turn.
            return None, error_msg

        # 2. Extract and print reasoning (text parts)
        model_content = resp.candidates[0].content
        found_text = False
        text_content = ""
        if model_content.parts:
            for part in model_content.parts:
                text_val = getattr(part, 'text', None) or getattr(part, 'thought', None)
                if text_val:
                    if verbose_llm:
                        print(f"\n{Colors.OKBLUE}[THOUGHT]{Colors.ENDC} {text_val.strip()}\n")
                    text_content += text_val
                    found_text = True
        
        # 3. Handle Tool Calls
        if hasattr(resp, 'function_calls') and resp.function_calls:
            messages.append({"role": "assistant", "model_content": model_content})
            
            num_calls = len(resp.function_calls)
            print(f"{Colors.OKCYAN}[INFO] Model returned {num_calls} function call(s).{Colors.ENDC}")

            # Gemini requires all N function responses in ONE Content with N parts.
            # Collect parts here, then append a single bundled message after the loop.
            response_parts = []
            done_early = False

            for i, function_call in enumerate(resp.function_calls):
                name = function_call.name
                args = {k: v for k, v in function_call.args.items()}
                
                # REASONING EXTRACTION: If the tool has a 'thought' argument, print it as [THOUGHT]
                if 'thought' in args:
                    print(f"\n{Colors.OKBLUE}[THOUGHT]{Colors.ENDC} {args['thought'].strip()}\n")
                
                # Always print the tool call for visibility
                clean_args = {k: v for k, v in args.items() if k != 'thought'}
                print(f"{Colors.OKCYAN}[TOOL CALL {i+1}/{num_calls}] {name}({clean_args}){Colors.ENDC}")
                
                result = registry.execute_tool(name, args)
                
                # Always print the result summary
                success_color = Colors.OKGREEN if result.get('success') else Colors.FAIL
                match_count = result.get('match_count', result.get('matches', 0))
                print(f"{success_color}[TOOL OUTPUT {i+1}/{num_calls}] success={result.get('success')} matches={match_count}{Colors.ENDC}")
                
                # If there are sample matches (from filter_dataset), print them for visibility
                if 'sample_matches' in result and result['sample_matches']:
                    print(f"{Colors.OKBLUE}[SAMPLE MATCHES]{Colors.ENDC}")
                    for j, m in enumerate(result['sample_matches']):
                        raw = m.get('_raw', str(m))
                        # Truncate for display
                        display_raw = (raw[:150] + '...') if len(raw) > 150 else raw
                        print(f"  {j+1}. {display_raw}")
                    print()
                if verbose_llm:
                    print(f"{Colors.OKGREEN}[DEBUG] Full Tool Output: {result}{Colors.ENDC}")
                
                response_parts.append(
                    _make_tool_result(function_call, name, result)
                )
                
                if registry.is_done:
                    done_early = True
                    break

            # Append all responses as a single bundled message
            messages.append({"role": "function_responses", "parts": response_parts})

            if done_early:
                candidate_rule = registry.finalized_rule
                print(f"{Colors.OKGREEN}[PASS] Rule finalized through tool call.{Colors.ENDC}")
                registry.cleanup_transient_datasets()
                return candidate_rule, "Success"
        else:
            # It sent only raw text
            messages.append({"role": "assistant", "content": text_content if found_text else str(resp)})
            
            if verbose_llm and not found_text:
                 # Fallback print if for some reason text extraction failed but object exists
                 print(f"\n{Colors.OKBLUE}--- [VERBOSE] Raw LLM Output ---{Colors.ENDC}\n{resp}\n")
            
            messages.append({"role": "user", "content": "You provided reasoning but did not call a tool. Please proceed to call `test_rule` or `submit_rule` now."})
            
    # Fallback logic: If we exceeded steps but have a passing test, use that rule as the submission.
    if not registry.is_done and registry.test_passed:
        print(f"\n{Colors.WARNING}[WARN] Agent reached max steps ({max_agent_steps}) without submitting, but has a passing test. Falling back to the last successful rule.{Colors.ENDC}")
        registry.finalized_rule = registry.last_tested_rule
        registry.is_done = True
        return registry.finalized_rule, "Success (Fallback from max steps)"

    return None, f"Rule Generation Failed: Exceeded {max_agent_steps} agent steps."

def validate_discovery_results(
    orig_log: str,
    discovery_results: List[Dict[str, Any]],
    api_key: str,
    sample_size: int = 5,
    vertexai: bool = False,
    vertex_project: str = os.environ.get("VERTEX_PROJECT", ""),
) -> List[Dict[str, Any]]:
    """Samples discovery results and asks the LLM to validate them to reduce false positives."""
    if not discovery_results:
        return []

    import random
    sample = random.sample(discovery_results, min(len(discovery_results), sample_size))
    candidate_lines = [res.get("_raw", "N/A") for res in sample]

    print(f"{Colors.OKBLUE}[INFO] Validating {len(sample)} discovery samples for false positives...{Colors.ENDC}")

    messages = [
        {"role": "system", "content": prompts.get_system_role()},
        {"role": "user", "content": prompts.get_discovery_validation_prompt(orig_log, candidate_lines)}
    ]
 
    try:
        resp_text = call_llm(messages, api_key, response_format={"type": "json_object"}, vertexai=vertexai, vertex_project=vertex_project).text
        eval_data = json.loads(resp_text)
        evaluations = eval_data.get("evaluations", [])

        for i, eval_item in enumerate(evaluations):
            verdict = eval_item.get("verdict", "Unknown")
            reason = eval_item.get("reason", "N/A")
            color = Colors.OKGREEN if "Legitimate" in verdict else Colors.WARNING
            print(f"  Sample {i+1}: {color}{verdict}{Colors.ENDC} - {reason}")

        return evaluations
    except Exception as e:
        print(f"{Colors.WARNING}[WARN] Discovery validation failed: {e}{Colors.ENDC}")
        return []


def run_harness(
    log_line: str,
    repo_path: str,
    api_key: str,
    splunk_engine: SplunkEngine,
    sourcetype: Optional[str],
    index_name: Optional[str] = None,
    new_index: bool = False,
    verbose_llm: bool = False,
    description: Optional[str] = None,
    vertexai: bool = False,
    vertex_project: str = os.environ.get("VERTEX_PROJECT", ""),
    no_description: bool = False,
):
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}=== Splunk Rule Generation Harness ({MODEL_NAME}) ==={Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    if not sourcetype:
        print(f"{Colors.OKBLUE}[INFO] Sourcetype not provided. Inferring via LLM...{Colors.ENDC}")
        sourcetype = infer_sourcetype(log_line, api_key, vertexai=vertexai, vertex_project=vertex_project)
        print(f"{Colors.OKGREEN}[INFO] Inferred Sourcetype: {sourcetype}{Colors.ENDC}")

    if not description:
        print(f"{Colors.WARNING}[WARN] No description provided. LLM will rely solely on the raw log.{Colors.ENDC}")

    GT_INDEX = "ground_truth_index"

    if new_index:
        if index_name:
            DISCO_INDEX = index_name
            print(f"{Colors.OKBLUE}[INFO] Deleting existing index '{DISCO_INDEX}' for fresh setup...{Colors.ENDC}")
            splunk_engine.delete_store(DISCO_INDEX)
            time.sleep(2)
        else:
            DISCO_INDEX = f"default_index_{int(time.time())}"
 
        print(f"{Colors.OKBLUE}[INFO] Setting up new repository index '{DISCO_INDEX}'...{Colors.ENDC}")
        splunk_engine.create_store(DISCO_INDEX)
        time.sleep(2)
    else:
        DISCO_INDEX = index_name
        print(f"{Colors.OKBLUE}[INFO] Using existing repository index '{DISCO_INDEX}'{Colors.ENDC}")
 
    splunk_engine.create_store(GT_INDEX)
    time.sleep(1)
 
    print(f"\n{Colors.OKBLUE}[STEP 1] Ingesting ground truth log line...{Colors.ENDC}")
    if not splunk_engine.ingest_line(log_line, GT_INDEX, sourcetype, "ciscoasa"):
        print(f"{Colors.FAIL}Aborting: Ground truth ingestion failed.{Colors.ENDC}")
        record_query_history(log_line, None, 0, "Ingestion Failed")
        return
 
    validated_rule, status = generate_rule_for_log(
        log_line,
        GT_INDEX,
        api_key,
        splunk_engine=splunk_engine,
        verbose_llm=verbose_llm,
        description=description,
        vertexai=vertexai,
        vertex_project=vertex_project,
        main_dataset=DISCO_INDEX,
        no_description=no_description,
    )
    if not validated_rule:
        print(f"{Colors.FAIL}Aborting: {status}{Colors.ENDC}")
        record_query_history(log_line, None, 0, status)
        return

    if new_index:
        print(f"\n{Colors.OKBLUE}[STEP 3] Ingesting repository logs from '{repo_path}' into '{DISCO_INDEX}'...{Colors.ENDC}")
        if not os.path.exists(repo_path):
            print(f"{Colors.FAIL}Error: '{repo_path}' does not exist.{Colors.ENDC}")
            return

        files_to_ingest = []
        if os.path.isfile(repo_path):
            files_to_ingest.append(repo_path)
        else:
            for filename in os.listdir(repo_path):
                file_path = os.path.join(repo_path, filename)
                if os.path.isfile(file_path) and not filename.startswith("."):
                    files_to_ingest.append(file_path)

        for file_path in files_to_ingest:
            print(f" - Ingesting {os.path.basename(file_path)}...")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [l.strip() for l in f]
                splunk_engine.ingest_lines(lines, DISCO_INDEX, sourcetype, "ciscoasa")
            except Exception as e:
                print(f"{Colors.WARNING}[WARN] Failed to read {file_path}: {e}{Colors.ENDC}")

        print(f"{Colors.OKBLUE}[INFO] Waiting for Splunk to index repository logs...{Colors.ENDC}")
        time.sleep(5)
    else:
        print(f"\n{Colors.OKBLUE}[STEP 3] Skipping repository ingestion (--new-index is off){Colors.ENDC}")

    print(f"\n{Colors.OKBLUE}[STEP 4] Discovering other matches...{Colors.ENDC}")
 
    _, _, disco_count = splunk_engine.test_pattern(f"index={DISCO_INDEX}")
    if disco_count == 0:
        print(f"{Colors.WARNING}[WARN] Repository index '{DISCO_INDEX}' appears to be empty! Discovery will not find anything.{Colors.ENDC}")

    discovery_spl = validated_rule.strip()
    if "index=" in discovery_spl:
        pattern = rf'index\s*=\s*["\']?{GT_INDEX}["\']?'
        discovery_spl = re.sub(pattern, f"index={DISCO_INDEX}", discovery_spl)
    else:
        if discovery_spl.startswith("search"):
            discovery_spl = discovery_spl.replace("search", f"search index={DISCO_INDEX} ", 1)
        elif discovery_spl.startswith("|"):
             if "tstats" in discovery_spl.lower() and "where" in discovery_spl.lower():
                discovery_spl = re.sub(r'where\s+', f'where index={DISCO_INDEX} ', discovery_spl, count=1, flags=re.IGNORECASE)
             elif "tstats" in discovery_spl.lower():
                discovery_spl = re.sub(r'from\s+datamodel=[\w.]+', lambda m: f"{m.group(0)} where index={DISCO_INDEX}", discovery_spl, count=1, flags=re.IGNORECASE)
        else:
            discovery_spl = f"search index={DISCO_INDEX} {discovery_spl}"

    if not discovery_spl.strip().startswith("|") and not discovery_spl.strip().startswith("search"):
        discovery_spl = f"search {discovery_spl}"
    print(f"Running Discovery SPL: {Colors.OKBLUE}{discovery_spl}{Colors.ENDC}")
    success, msg, disco_hits = splunk_engine.test_pattern(discovery_spl)
 
    sample_matches = []
    if success and disco_hits > 0:
        print(f"{Colors.OKGREEN}[PASS] Discovery found {disco_hits} matches.{Colors.ENDC}")
        sample_results = splunk_engine.get_matches(discovery_spl, limit=10)
 
        if sample_results:
            sample_matches = sample_results
            print(f"\n{Colors.HEADER}--- Discovery Validation Sample ---{Colors.ENDC}")
            validate_discovery_results(log_line, sample_results, api_key, vertexai=vertexai, vertex_project=vertex_project)
    elif not success:
        print(f"{Colors.WARNING}[FAIL] Discovery SPL failed: {msg}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}[INFO] Discovery found no other matches.{Colors.ENDC}")

    status = "Success" if (validated_rule and success) else "Discovery Failed"
    record_query_history(log_line, validated_rule, disco_hits, status, sample_matches)

    return


# -----------------------------------------------------------------------------
# Regex-mode harness
# -----------------------------------------------------------------------------

def run_harness_regex(
    log_line: str,
    repo_path: Optional[str],
    api_key: str,
    verbose_llm: bool = False,
    description: Optional[str] = None,
    vertexai: bool = False,
    vertex_project: str = os.environ.get("VERTEX_PROJECT", ""),
    no_description: bool = False,
):
    """End-to-end harness using regex patterns instead of Splunk/SPL."""
    engine = RegexEngine()

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}=== Regex Rule Generation Harness ({MODEL_NAME}) ==={Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    if not description:
        print(f"{Colors.WARNING}[WARN] No description provided. LLM will rely solely on the raw log.{Colors.ENDC}")

    GT_STORE = "ground_truth"
    engine.create_store(GT_STORE)
    engine.ingest_line(log_line, GT_STORE)

    print(f"\n{Colors.OKBLUE}[STEP 1] Ground truth log loaded into memory.{Colors.ENDC}")

    validated_rule, status = generate_rule_for_log(
        log_line,
        GT_STORE,
        api_key,
        verbose_llm=verbose_llm,
        description=description,
        regex_engine=engine,
        vertexai=vertexai,
        vertex_project=vertex_project,
        main_dataset=DISCO_STORE,
        no_description=no_description,
    )
    if not validated_rule:
        print(f"{Colors.FAIL}Aborting: {status}{Colors.ENDC}")
        record_query_history(log_line, None, 0, status)
        return

    # Discovery phase
    DISCO_STORE = "discovery"
    engine.create_store(DISCO_STORE)

    if repo_path:
        print(f"\n{Colors.OKBLUE}[STEP 3] Loading repository logs from '{repo_path}'...{Colors.ENDC}")
        if not os.path.exists(repo_path):
            print(f"{Colors.FAIL}Error: '{repo_path}' does not exist.{Colors.ENDC}")
            return

        files_to_load: List[str] = []
        if os.path.isfile(repo_path):
            files_to_load.append(repo_path)
        else:
            for filename in os.listdir(repo_path):
                file_path = os.path.join(repo_path, filename)
                if os.path.isfile(file_path) and not filename.startswith("."):
                    files_to_load.append(file_path)

        for file_path in files_to_load:
            print(f" - Loading {os.path.basename(file_path)}...")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [l.strip() for l in f]
                engine.ingest_lines(lines, DISCO_STORE)
            except Exception as e:
                print(f"{Colors.WARNING}[WARN] Failed to read {file_path}: {e}{Colors.ENDC}")

        print(f"{Colors.OKBLUE}[INFO] {engine.count(DISCO_STORE)} lines loaded into discovery store.{Colors.ENDC}")
    else:
        print(f"\n{Colors.OKBLUE}[STEP 3] No --repo-path provided; skipping discovery.{Colors.ENDC}")

    print(f"\n{Colors.OKBLUE}[STEP 4] Discovering other matches...{Colors.ENDC}")
    print(f"Running discovery regex: {Colors.OKBLUE}{validated_rule}{Colors.ENDC}")
    success, msg, disco_hits = engine.test_pattern(validated_rule, DISCO_STORE)

    sample_matches: List[Dict[str, Any]] = []
    if success and disco_hits > 0:
        print(f"{Colors.OKGREEN}[PASS] Discovery found {disco_hits} matches.{Colors.ENDC}")
        sample_results = engine.get_matches(validated_rule, DISCO_STORE, limit=10)
        if sample_results:
            sample_matches = sample_results
            print(f"\n{Colors.HEADER}--- Discovery Validation Sample ---{Colors.ENDC}")
            validate_discovery_results(log_line, sample_results, api_key, vertexai=vertexai, vertex_project=vertex_project)
    elif not success:
        print(f"{Colors.WARNING}[FAIL] Discovery regex failed: {msg}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}[INFO] Discovery found no other matches.{Colors.ENDC}")

    status = "Success" if (validated_rule and success) else "Discovery Failed"
    record_query_history(log_line, validated_rule, disco_hits, status, sample_matches)


def run_harness_python(
    log_line: str,
    repo_path: Optional[str],
    api_key: str,
    verbose_llm: bool = False,
    description: Optional[str] = None,
    vertexai: bool = False,
    vertex_project: str = os.environ.get("VERTEX_PROJECT", ""),
    no_description: bool = False,
):
    """End-to-end harness using Python scripts inside Docker."""
    engine = PythonEngine()

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}=== Python Rule Generation Harness ({MODEL_NAME}) ==={Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    if not description:
        print(f"{Colors.WARNING}[WARN] No description provided. LLM will rely solely on the raw log.{Colors.ENDC}")

    GT_STORE = "ground_truth"
    engine.create_store(GT_STORE)
    engine.ingest_line(log_line, GT_STORE)

    print(f"\n{Colors.OKBLUE}[STEP 1] Ground truth log loaded into memory.{Colors.ENDC}")

    validated_rule, status = generate_rule_for_log(
        log_line,
        GT_STORE,
        api_key,
        verbose_llm=verbose_llm,
        description=description,
        python_engine=engine,
        vertexai=vertexai,
        vertex_project=vertex_project,
        main_dataset=DISCO_STORE,
        no_description=no_description,
    )
    if not validated_rule:
        print(f"{Colors.FAIL}Aborting: {status}{Colors.ENDC}")
        record_query_history(log_line, None, 0, status)
        return

    # Discovery phase
    DISCO_STORE = "discovery"
    engine.create_store(DISCO_STORE)

    if repo_path:
        print(f"\n{Colors.OKBLUE}[STEP 3] Loading repository logs from '{repo_path}'...{Colors.ENDC}")
        if not os.path.exists(repo_path):
            print(f"{Colors.FAIL}Error: '{repo_path}' does not exist.{Colors.ENDC}")
            return

        files_to_load: List[str] = []
        if os.path.isfile(repo_path):
            files_to_load.append(repo_path)
        else:
            for filename in os.listdir(repo_path):
                file_path = os.path.join(repo_path, filename)
                if os.path.isfile(file_path) and not filename.startswith("."):
                    files_to_load.append(file_path)

        for file_path in files_to_load:
            print(f" - Loading {os.path.basename(file_path)}...")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [l.strip() for l in f]
                engine.ingest_lines(lines, DISCO_STORE)
            except Exception as e:
                print(f"{Colors.WARNING}[WARN] Failed to read {file_path}: {e}{Colors.ENDC}")

        print(f"{Colors.OKBLUE}[INFO] {engine.count(DISCO_STORE)} lines loaded into discovery store.{Colors.ENDC}")
    else:
        print(f"\n{Colors.OKBLUE}[STEP 3] No --repo-path provided; skipping discovery.{Colors.ENDC}")

    print(f"\n{Colors.OKBLUE}[STEP 4] Discovering other matches...{Colors.ENDC}")
    print(f"Running discovery python script: {Colors.OKBLUE}check_log() function{Colors.ENDC}")
    success, msg, disco_hits = engine.test_pattern(validated_rule, DISCO_STORE)

    sample_matches: List[Dict[str, Any]] = []
    if success and disco_hits > 0:
        print(f"{Colors.OKGREEN}[PASS] Discovery found {disco_hits} matches.{Colors.ENDC}")
        sample_results = engine.get_matches(validated_rule, DISCO_STORE, limit=10)
        if sample_results:
            sample_matches = sample_results
            print(f"\n{Colors.HEADER}--- Discovery Validation Sample ---{Colors.ENDC}")
            validate_discovery_results(log_line, sample_results, api_key, vertexai=vertexai, vertex_project=vertex_project)
    elif not success:
        print(f"{Colors.WARNING}[FAIL] Discovery python failed: {msg}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}[INFO] Discovery found no other matches.{Colors.ENDC}")

    status = "Success" if (validated_rule and success) else "Discovery Failed"
    record_query_history(log_line, validated_rule, disco_hits, status, sample_matches)


def _resolve_api_key(args) -> str:
    """Return the Gemini API key from args or well-known file locations."""
    gemini_key = args.gemini_api_key
    if not gemini_key:
        possible_paths = [
            "../gemini.txt",
            "../../gemini.txt",
            os.path.join(os.path.dirname(__file__), "..", "gemini.txt"),
            os.path.join(os.path.dirname(__file__), "..", "..", "gemini.txt"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    gemini_key = f.read().strip()
                break
    if not gemini_key:
        print(f"{Colors.FAIL}Error: Gemini API Key not found in gemini.txt or provided via argument.{Colors.ENDC}")
        sys.exit(1)
    return gemini_key


def _resolve_target_log(args) -> str:
    """Return the target log line from --log-line or --log-file."""
    target_log = args.log_line
    if args.log_file:
        try:
            with open(args.log_file, 'r') as f:
                target_log = f.read().strip()
        except Exception as e:
            print(f"Error reading log file: {e}")
            sys.exit(1)

    lines = [l for l in target_log.splitlines() if l.strip()]
    if len(lines) > 1:
        print(f"{Colors.WARNING}[WARN] Target log contains {len(lines)} lines. "
              f"This agent is optimized for a single suspicious log line. "
              f"Using the full block as input.{Colors.ENDC}")
    return target_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Input group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--log-line", help="A single suspicious log line string")
    input_group.add_argument("--log-file", help="Path to file containing one log line")

    # Mode group
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--splunk-mode", action="store_true",
                           help="Use Splunk/SPL generation and testing (requires Splunk instance)")
    mode_group.add_argument("--regex-mode", action="store_true",
                           help="Use regex matching instead of Splunk/SPL (no Splunk instance needed)")
    mode_group.add_argument("--python-mode", action="store_true",
                           help="Use Python script matching inside Docker (no Splunk instance needed)")

    parser.add_argument("--sourcetype", help="Sourcetype for logs (Splunk mode only, LLM inferred if not provided)")
    parser.add_argument("--repo-path", help="Path to logs for ingestion/discovery")
    parser.add_argument("--splunk-url", default="https://localhost:8089")
    parser.add_argument("--splunk-user", default="admin")
    parser.add_argument("--splunk-pass", default="password")
    parser.add_argument("--gemini-api-key", help="Gemini API Key (overrides file lookup)")
    parser.add_argument("--vertexai", action="store_true", help="Use Google Vertex AI instead of standard API call")
    parser.add_argument("--vertexai-project", default=os.environ.get("VERTEX_PROJECT", ""),
                        help="Vertex AI Project ID (default: $VERTEX_PROJECT)")
    parser.add_argument("--index-name", help="Name of the Splunk index to use for discovery (Splunk mode only)")
    parser.add_argument("--new-index", action="store_true",
                        help="Create a new index (or clear existing one) and ingest logs from --repo-path (Splunk mode only)")
    parser.add_argument("--verbose-llm", action="store_true", help="Print the raw LLM output to stdout")
    parser.add_argument("--description", help="A brief human-written description of why the provided log line is suspicious")
    parser.add_argument("--no-description", action="store_true", help="Do not pass the log description to the LLM")
    args = parser.parse_args()
 
    gemini_key = _resolve_api_key(args) if not args.vertexai else ""
    target_log = _resolve_target_log(args)
 
    if args.python_mode:
        run_harness_python(
            target_log,
            args.repo_path,
            gemini_key,
            verbose_llm=args.verbose_llm,
            description=args.description,
            vertexai=args.vertexai,
            vertex_project=args.vertexai_project,
            no_description=args.no_description,
        )
    elif args.regex_mode:
        run_harness_regex(
            target_log,
            args.repo_path,
            gemini_key,
            verbose_llm=args.verbose_llm,
            description=args.description,
            vertexai=args.vertexai,
            vertex_project=args.vertexai_project,
            no_description=args.no_description,
        )
    elif args.splunk_mode:
        if not args.new_index and not args.index_name:
            print(f"{Colors.FAIL}Error: --index-name is required when not using --new-index (Splunk mode).{Colors.ENDC}")
            sys.exit(1)
        if args.new_index and not args.repo_path:
            print(f"{Colors.FAIL}Error: --repo-path is required when using --new-index.{Colors.ENDC}")
            sys.exit(1)
 
        splunk_engine = SplunkEngine(args.splunk_url, args.splunk_user, args.splunk_pass)
 
        run_harness(
            target_log,
            args.repo_path,
            gemini_key,
            splunk_engine,
            args.sourcetype,
            index_name=args.index_name,
            new_index=args.new_index,
            verbose_llm=args.verbose_llm,
            description=args.description,
            vertexai=args.vertexai,
            vertex_project=args.vertexai_project,
            no_description=args.no_description,
        )
