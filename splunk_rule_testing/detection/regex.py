"""
Regex-based detection proxy.

Provides an in-memory log store and regex matching engine as an alternative
to the Splunk REST API backend.  No network calls or external services needed.
"""

import re
from typing import Dict, List, Set, Tuple, Any


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class RegexEngine:
    """In-memory log store with regex pattern matching.

    Mirrors the Splunk proxy workflow: create a store (analogous to a Splunk
    index), ingest log lines, then test regex patterns against them.
    """

    def __init__(self) -> None:
        self._stores: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def create_store(self, name: str) -> None:
        """Create (or reset) an in-memory log store."""
        self._stores[name] = []

    def delete_store(self, name: str) -> None:
        """Delete an in-memory log store if it exists."""
        self._stores.pop(name, None)

    def list_stores(self) -> List[str]:
        return list(self._stores.keys())

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_line(self, line: str, store: str) -> None:
        """Add a single log line to a store (auto-creates if needed)."""
        if store not in self._stores:
            self._stores[store] = []
        stripped = line.strip()
        if stripped:
            self._stores[store].append(stripped)

    def ingest_lines(self, lines: List[str], store: str) -> int:
        """Add multiple log lines to a store. Returns the count ingested."""
        if store not in self._stores:
            self._stores[store] = []
        count = 0
        for line in lines:
            stripped = line.strip() if line else ""
            if stripped:
                self._stores[store].append(stripped)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _compile(self, pattern: str) -> re.Pattern:
        """Compile *pattern* with IGNORECASE + DOTALL."""
        return re.compile(pattern, re.IGNORECASE | re.DOTALL)

    def test_pattern(
        self, pattern: str, store: str
    ) -> Tuple[bool, str, int]:
        """Test a regex against every line in *store*.

        Returns ``(success, message, hit_count)``.
        *success* is ``False`` when the store is missing or the pattern is
        invalid; ``True`` otherwise (even when hit_count is 0).
        """
        if store not in self._stores:
            return False, f"Store '{store}' not found", 0
        try:
            compiled = self._compile(pattern)
        except re.error as exc:
            return False, f"Invalid regex: {exc}", 0
        count = sum(1 for line in self._stores[store] if compiled.search(line))
        return True, "OK", count

    def get_matches(
        self, pattern: str, store: str, limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """Return matching lines as dicts with a ``_raw`` key.

        Keeps the ``_raw`` key so downstream report logic works identically
        for both Splunk and regex modes.
        """
        if store not in self._stores:
            return []
        try:
            compiled = self._compile(pattern)
        except re.error:
            return []
        results: List[Dict[str, Any]] = []
        for line in self._stores[store]:
            if compiled.search(line):
                results.append({"_raw": line})
                if len(results) >= limit:
                    break
        return results

    def get_matched_raws(
        self, pattern: str, store: str, limit: int = 10000
    ) -> Set[str]:
        """Return the set of matching raw log lines."""
        if store not in self._stores:
            return set()
        try:
            compiled = self._compile(pattern)
        except re.error:
            return set()
        matched: Set[str] = set()
        for line in self._stores[store]:
            if compiled.search(line):
                matched.add(line)
                if len(matched) >= limit:
                    break
        return matched

    def count(self, store: str) -> int:
        """Return the number of lines currently held in *store*."""
        return len(self._stores.get(store, []))

    def get_store_content(self, store: str) -> Set[str]:
        """Return all log lines currently in *store* as a set."""
        return set(self._stores.get(store, []))
