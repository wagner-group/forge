"""OpenAI (GPT-5.6 family) backend for the FORGE agent loop.

Mirrors the Gemini call surface in agent.py so `generate_rule_for_log` works
unchanged apart from where it builds tool-result parts.

Design notes
------------
* Uses the **Responses API**, which is the recommended path for GPT-5.6
  tool-calling and is the only one that exposes the full `reasoning.effort`
  range and returns reasoning items you can carry across turns.
* `reasoning.effort` is ALWAYS sent explicitly. Omitting it bills ~1.5x more
  than sending `"none"` for identical results, and `"none"` emits no reasoning
  items -- which also sidesteps the reported `usage.output_tokens` re-summing
  inflation that scales with reasoning-item count.
* Gemini's `mode="ANY"` + `allowed_function_names` maps to
  `tool_choice={"type": "allowed_tools", "mode": "required", ...}`. Some models
  reject `allowed_tools`; we fall back to plain `"required"` and rely on the
  registry to reject out-of-state calls.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_EFFORT = "none"

# Retryable substrings, mirroring agent.py's Gemini retry policy.
_RETRYABLE = ("429", "500", "502", "503", "rate_limit", "overloaded",
              "timeout", "Connection", "server_error")


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

class UsageTracker:
    """Accumulates token usage across one trial (reset per trial, like the call counter)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.calls = 0

    def record(self, usage: Any) -> None:
        if usage is None:
            return
        self.calls += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        in_det = getattr(usage, "input_tokens_details", None)
        if in_det is not None:
            self.cached_tokens += getattr(in_det, "cached_tokens", 0) or 0
        out_det = getattr(usage, "output_tokens_details", None)
        if out_det is not None:
            self.reasoning_tokens += getattr(out_det, "reasoning_tokens", 0) or 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "api_calls": self.calls,
        }


USAGE = UsageTracker()


# ---------------------------------------------------------------------------
# Response shim -- exposes the handful of attributes the agent loop reads
# ---------------------------------------------------------------------------

class _Part:
    """Mimics a Gemini text part."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.thought = None


class _ToolCall:
    """Mimics a Gemini FunctionCall (`.name`, `.args`) and carries the call_id."""

    def __init__(self, name: str, args: Dict[str, Any], call_id: str) -> None:
        self.name = name
        self.args = args
        self.call_id = call_id


class _ModelContent:
    """Replayable assistant turn: keeps the raw Responses output items."""

    def __init__(self, items: List[Any], parts: List[_Part]) -> None:
        self.items = items      # raw output items, replayed verbatim next turn
        self.parts = parts      # text parts, for the loop's reasoning printout


class _Candidate:
    def __init__(self, content: _ModelContent, finish_reason: Optional[str]) -> None:
        self.content = content
        self.finish_reason = finish_reason


class OpenAIResponse:
    """Duck-types the Gemini response object consumed by generate_rule_for_log."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        items = list(getattr(raw, "output", []) or [])

        parts: List[_Part] = []
        calls: List[_ToolCall] = []
        for item in items:
            itype = getattr(item, "type", None)
            if itype == "message":
                for block in getattr(item, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(_Part(text))
            elif itype == "function_call":
                try:
                    args = json.loads(getattr(item, "arguments", "") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(_ToolCall(
                    name=getattr(item, "name", ""),
                    args=args if isinstance(args, dict) else {},
                    call_id=getattr(item, "call_id", ""),
                ))

        content = _ModelContent(items, parts)
        self.candidates = [_Candidate(content, getattr(raw, "status", None))]
        self.function_calls = calls
        self.usage = getattr(raw, "usage", None)


def make_tool_result(tool_call: _ToolCall, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build the Responses `function_call_output` item for one executed tool call."""
    return {
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": json.dumps(result, default=str),
    }


# ---------------------------------------------------------------------------
# Tool schemas -- explicit, mirroring ToolRegistry's signatures and docstrings
# ---------------------------------------------------------------------------

def build_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "test_rule",
            "description": (
                "Test a candidate detection rule against the suspicious input log. "
                "You MUST use this tool to verify your logic before submitting. "
                "Check the success field in the returned response to see if it worked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "A brief explanation of your analysis and strategy for this test."},
                    "rule": {"type": "string", "description": "The detection rule code (Python, Splunk SPL, or Regex depending on the mode)."},
                },
                "required": ["thought", "rule"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "filter_dataset",
            "description": (
                "Apply a detection rule to an existing dataset to find matching logs. "
                "Use this to discover other instances of the behavior in the background "
                "\"main\" dataset, verify robustness, or find potential False Positives. "
                "Up to 10 random matches are returned as a sample of the coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Your reasoning for filtering this dataset."},
                    "rule": {"type": "string", "description": "The detection rule (regex, SPL, or Python) to apply."},
                    "dataset": {"type": "string", "description": "Dataset to filter. Use \"main\" for the primary log repository."},
                    "new_dataset": {
                        "type": ["string", "null"],
                        "description": "Optional name for a new transient dataset to store the matches. Cannot be \"main\" or an existing name.",
                    },
                },
                "required": ["thought", "rule", "dataset", "new_dataset"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "submit_rule",
            "description": (
                "Submit your finalized, tested detection rule. Omit final_rule to submit "
                "the rule that most recently passed test_rule()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "A final summary of why this rule is ready for submission."},
                    "final_rule": {
                        "type": ["string", "null"],
                        "description": "Optional. The rule code to submit. If null, uses the last successful test.",
                    },
                },
                "required": ["thought", "final_rule"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def _to_responses_input(messages: List[Dict[str, Any]]) -> tuple[str, List[Any]]:
    """Split FORGE's message list into (instructions, Responses input items)."""
    instructions = ""
    items: List[Any] = []

    for m in messages:
        role = m.get("role")
        if role == "system":
            instructions += m["content"] + "\n"
        elif role == "user":
            items.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            if "model_content" in m:
                # Replay the exact output items (reasoning + function_call) verbatim.
                items.extend(m["model_content"].items)
            else:
                items.append({"role": "assistant", "content": m.get("content", "")})
        elif role == "function_responses":
            # Already-built function_call_output dicts, one per executed call.
            items.extend(m["parts"])
        elif role == "function":
            items.append(m["function_response"])

    return instructions, items


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def call_llm(
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    tools: Optional[List[Dict[str, Any]]] = None,
    allowed_tools: Optional[List[str]] = None,
    max_output_tokens: int = 16384,
    max_retries: int = 6,
) -> OpenAIResponse:
    """One Responses API turn. Returns a Gemini-shaped shim."""
    try:
        import httpx
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("The 'openai' package is required. pip install openai")

    # 600s was too generous to be useful as a stall detector. A normal call
    # completes in ~31s, so a 10-minute ceiling meant a wedged connection sat
    # undetected; observed stalls ran 4-5 hours with the socket ESTABLISHED and
    # no data flowing, which suggests the read timeout was not being reached at
    # all. A tighter budget plus explicit client-level retries turns that into a
    # few minutes of lost work instead of an overnight hang.
    #
    # max_retries is now passed to the client. It was previously only this
    # function's own retry counter, so the SDK silently used its default of 2.
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(connect=20.0, read=180.0, write=60.0, pool=20.0),
        max_retries=max_retries,
    )

    instructions, input_items = _to_responses_input(messages)

    kwargs: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
        # Always explicit: omitting effort carries a ~1.5x cost penalty.
        "reasoning": {"effort": effort},
        "store": False,
    }
    if tools:
        kwargs["tools"] = tools
        if allowed_tools:
            kwargs["tool_choice"] = {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": n} for n in allowed_tools],
            }
        else:
            kwargs["tool_choice"] = "required"

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = client.responses.create(**kwargs)
            USAGE.record(getattr(raw, "usage", None))
            return OpenAIResponse(raw)
        except Exception as e:
            msg = str(e)
            # allowed_tools is not accepted by every model; degrade once and retry.
            if "allowed_tools" in msg and kwargs.get("tool_choice", {}) != "required":
                kwargs["tool_choice"] = "required"
                continue
            last_error = e
            if not any(code in msg for code in _RETRYABLE) or attempt == max_retries:
                break
            time.sleep(2 ** attempt * 5)

    raise Exception(f"LLM Call failed: {last_error}")
