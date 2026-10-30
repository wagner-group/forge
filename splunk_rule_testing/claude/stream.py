"""Parse Claude Code stream-json output for live verbose eval."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TextIO

from detection import Colors

_TOOL_SUMMARY_MAX = 200
_THOUGHT_SUMMARY_MAX = 500


def extract_final_result(combined_log: str) -> Optional[Dict[str, Any]]:
    """Return the last stream-json event with type=result from accumulated log text."""
    result: Optional[Dict[str, Any]] = None
    for line in combined_log.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    return result


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _tool_summary(name: str, tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return _truncate(str(tool_input), _TOOL_SUMMARY_MAX)
    if name == "Bash":
        cmd = tool_input.get("command") or tool_input.get("cmd") or str(tool_input)
        return _truncate(str(cmd), _TOOL_SUMMARY_MAX)
    if name in ("Read", "Write", "Edit"):
        path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("file")
        if path:
            return str(path)
    return _truncate(json.dumps(tool_input, ensure_ascii=False), _TOOL_SUMMARY_MAX)


def _try_parse_json(raw: str) -> Optional[Any]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _tool_input_ready(name: str, tool_input: Any) -> bool:
    if tool_input is None:
        return False
    if isinstance(tool_input, dict):
        if not tool_input:
            return False
        if name == "Bash":
            return bool(tool_input.get("command") or tool_input.get("cmd"))
        if name in ("Read", "Write", "Edit"):
            return bool(
                tool_input.get("file_path")
                or tool_input.get("path")
                or tool_input.get("file")
            )
        return True
    return bool(str(tool_input).strip())


def _content_blocks(message: Any) -> List[Dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


class StreamEventPrinter:
    """Convert stream-json lines into human-readable eval output."""

    def __init__(self, *, max_turns: Optional[int] = None, out: Optional[TextIO] = None) -> None:
        self.max_turns = max_turns
        self.out = out
        self.turn_count = 0
        self._pending_thought = ""
        self._last_printed_thought = ""
        self._pending_tool_blocks: Dict[int, Dict[str, Any]] = {}
        self._printed_tool_ids: set = set()

    def _emit_tool(
        self,
        outputs: List[str],
        tool_id: Optional[str],
        name: str,
        tool_input: Any,
    ) -> None:
        if not _tool_input_ready(name, tool_input):
            return
        dedup_key = tool_id or f"{name}:{_tool_summary(name, tool_input)}"
        if dedup_key in self._printed_tool_ids:
            return
        self._printed_tool_ids.add(dedup_key)
        summary = _tool_summary(name, tool_input)
        outputs.append(f"{Colors.OKCYAN}[TOOL]{Colors.ENDC} {name}: {summary}")

    def _resolve_pending_tool(
        self,
        outputs: List[str],
        index: int,
        *,
        require_ready: bool = True,
    ) -> None:
        pending = self._pending_tool_blocks.get(index)
        if pending is None:
            return
        tool_input = pending.get("input")
        if tool_input is None:
            tool_input = _try_parse_json(pending.get("input_json") or "")
        if require_ready and not _tool_input_ready(pending["name"], tool_input):
            return
        self._pending_tool_blocks.pop(index, None)
        self._emit_tool(outputs, pending.get("id"), pending["name"], tool_input)

    def _append_thought(self, text: str) -> None:
        chunk = text
        if not chunk or not chunk.strip():
            return
        if self._last_printed_thought:
            stripped = chunk.strip()
            if stripped == self._last_printed_thought:
                return
            if stripped in self._last_printed_thought:
                return
        pending = self._pending_thought
        if not pending:
            self._pending_thought = chunk
            return
        if chunk in pending:
            return
        if pending in chunk:
            self._pending_thought = chunk
            return
        if pending.endswith(chunk):
            return
        if chunk.startswith(pending):
            self._pending_thought = chunk
            return
        self._pending_thought = pending + chunk

    def _emit_turn_outputs(self, outputs: List[str], turn_outputs: List[str]) -> None:
        if not turn_outputs:
            return
        self.turn_count += 1
        if self.max_turns is not None:
            turn_label = f"[TURN {self.turn_count}/{self.max_turns}]"
        else:
            turn_label = f"[TURN {self.turn_count}]"
        outputs.append(f"{Colors.HEADER}{turn_label}{Colors.ENDC}")
        outputs.extend(turn_outputs)

    def process_line(self, line: str) -> List[str]:
        line = line.strip()
        if not line:
            return []
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return [f"{Colors.WARNING}[STREAM]{Colors.ENDC} {_truncate(line, 200)}"]

        outputs: List[str] = []
        event_type = event.get("type")

        if event_type == "assistant":
            turn_outputs: List[str] = []
            message = event.get("message") or {}
            for block in _content_blocks(message):
                turn_outputs.extend(self._handle_content_block(block))
            self._flush_thought(turn_outputs)
            self._emit_turn_outputs(outputs, turn_outputs)

        elif event_type == "stream_event":
            nested = event.get("event") or {}
            nested_type = nested.get("type")
            if nested_type == "content_block_delta":
                delta = nested.get("delta") or {}
                index = nested.get("index")
                if delta.get("type") == "text_delta":
                    text = delta.get("text") or ""
                    if text:
                        self._append_thought(text)
                elif delta.get("type") == "thinking_delta":
                    text = delta.get("thinking") or delta.get("text") or ""
                    if text:
                        self._append_thought(text)
                elif delta.get("type") == "input_json_delta" and index is not None:
                    pending = self._pending_tool_blocks.get(index)
                    if pending is not None:
                        pending["input_json"] = (
                            pending.get("input_json", "") + delta.get("partial_json", "")
                        )
                        parsed = _try_parse_json(pending["input_json"])
                        if parsed is not None:
                            pending["input"] = parsed
                            self._resolve_pending_tool(outputs, index)
            elif nested_type == "content_block_start":
                block = nested.get("content_block") or {}
                if block.get("type") == "tool_use":
                    self._flush_thought(outputs)
                    index = nested.get("index")
                    if index is None:
                        index = len(self._pending_tool_blocks)
                    start_input = block.get("input")
                    self._pending_tool_blocks[index] = {
                        "name": block.get("name") or "tool",
                        "id": block.get("id"),
                        "input_json": "",
                        "input": start_input if _tool_input_ready(
                            block.get("name") or "tool", start_input
                        ) else None,
                    }
                    self._resolve_pending_tool(outputs, index)
            elif nested_type == "content_block_stop":
                index = nested.get("index")
                if index is not None:
                    self._resolve_pending_tool(outputs, index)

        elif event_type == "result":
            self._flush_thought(outputs)
            subtype = event.get("subtype") or "unknown"
            num_turns = event.get("num_turns")
            turn_info = f" ({num_turns} turns)" if num_turns is not None else ""
            outputs.append(
                f"{Colors.OKGREEN}[DONE]{Colors.ENDC} Claude Code finished: "
                f"{subtype}{turn_info}"
            )

        for text in outputs:
            if self.out is not None:
                print(text, file=self.out, flush=True)
            else:
                print(text, flush=True)
        return outputs

    def _handle_content_block(self, block: Dict[str, Any]) -> List[str]:
        outputs: List[str] = []
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text") or ""
            if text.strip():
                self._append_thought(text)
        elif block_type == "thinking":
            text = block.get("thinking") or block.get("text") or ""
            if text.strip():
                self._append_thought(text)
        elif block_type == "tool_use":
            self._flush_thought(outputs)
            name = block.get("name") or "tool"
            self._emit_tool(outputs, block.get("id"), name, block.get("input"))
        return outputs

    def _flush_thought(self, outputs: List[str]) -> None:
        thought = self._pending_thought.strip()
        self._pending_thought = ""
        if not thought or thought == self._last_printed_thought:
            return
        self._last_printed_thought = thought
        outputs.append(
            f"{Colors.OKBLUE}[THOUGHT]{Colors.ENDC} {_truncate(thought, _THOUGHT_SUMMARY_MAX)}"
        )
