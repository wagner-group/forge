"""
Python-based detection proxy with Docker isolation.

Provides an in-memory log store and executes LLM-generated Python scripts
inside a Docker container to ensure host-level security.
"""

import json
import subprocess
import os
from typing import Dict, List, Set, Tuple, Any


class PythonEngine:
    """In-memory log store with Docker-isolated Python execution.

    Mirrors the Splunk/Regex proxy workflow.
    """

    def __init__(self, image: str = "python:3.11-slim") -> None:
        self._stores: Dict[str, List[str]] = {}
        self._image = image

    def create_store(self, name: str) -> None:
        """Create (or reset) an in-memory log store."""
        self._stores[name] = []

    def delete_store(self, name: str) -> None:
        """Delete an in-memory log store if it exists."""
        self._stores.pop(name, None)

    def list_stores(self) -> List[str]:
        return list(self._stores.keys())

    def ingest_line(self, line: str, store: str) -> None:
        if store not in self._stores:
            self._stores[store] = []
        stripped = line.strip()
        if stripped:
            self._stores[store].append(stripped)

    def ingest_lines(self, lines: List[str], store: str) -> int:
        if store not in self._stores:
            self._stores[store] = []
        count = 0
        for line in lines:
            stripped = line.strip() if line else ""
            if stripped:
                self._stores[store].append(stripped)
                count += 1
        return count

    def _run_in_docker(self, code: str, logs: List[str]) -> Dict[str, Any]:
        """Executes the provided code against logs. Fallback to local execution if Docker is missing."""
        harness_script = """
import json
import sys
import re
import base64
import datetime
import math

def main():
    try:
        # Read the entire payload from stdin
        input_raw = sys.stdin.read()
        if not input_raw:
            return
        data = json.loads(input_raw)
        code = data['code']
        logs = data['logs']
        
        local_vars = {
            're': re,
            'json': json,
            'base64': base64,
            'datetime': datetime,
            'math': math
        }
        
        # Execute the LLM-provided code
        exec(code, {"__builtins__": __builtins__}, local_vars)
        check_func = local_vars.get('check_log')
        
        if not check_func:
            print(json.dumps({"error": "Function 'check_log(log_line: str) -> bool' not defined"}))
            return

        matched_raws = []
        for line in logs:
            try:
                # Call the custom detection function
                if check_func(line):
                    matched_raws.append(line)
            except Exception as e:
                # Don't fail the whole batch on one bad log
                continue
                
        # Return results to host as JSON
        print(json.dumps({"hits": len(matched_raws), "matched_raws": matched_raws}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
"""
        payload = json.dumps({"code": code, "logs": logs})
        
        # 1. Try Docker first
        try:
            cmd = [
                "docker", "run", "--rm", "-i", "--network", "none",
                self._image, "python", "-c", harness_script
            ]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=payload, timeout=30)
            
            if process.returncode == 0:
                return json.loads(stdout)
            
            # If Docker failed but is installed, return the error
            if "docker: command not found" not in stderr.lower() and "is not recognized as an internal or external command" not in stderr.lower():
                return {"error": f"Docker execution failed (exit {process.returncode}): {stderr}"}
                
        except (subprocess.TimeoutExpired, Exception):
            pass

        # 2. Fallback to local execution (Warning: Lower security)
        # We simulate the same harness logic locally
        import sys
        import io
        from contextlib import redirect_stdout
        
        if os.environ.get("FORGE_ALLOW_UNSANDBOXED_EXEC") != "1":
            return (False, "Docker unavailable and unsandboxed execution not permitted. "
                           "Python mode runs model-generated code; it requires Docker. "
                           "To run it on this host anyway, set "
                           "FORGE_ALLOW_UNSANDBOXED_EXEC=1 (not recommended). "
                           "Note: python mode produced none of the published results, "
                           "which are all regex mode.")
        try:
            print("[WARN] FORGE_ALLOW_UNSANDBOXED_EXEC=1: running model-generated code "
                  "on this host, outside Docker.")
            
            # Use a mock stdin/stdout to capture findings
            captured_stdout = io.StringIO()
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(payload)
            
            with redirect_stdout(captured_stdout):
                # We need to manually populate the environment for main() to see
                import re, base64, datetime, math
                
                # We re-implement the main loop here to avoid sys.stdin complexities in a shared process
                data = json.loads(payload)
                code_to_exec = data['code']
                logs_to_test = data['logs']
                
                local_vars = {
                    're': re,
                    'json': json,
                    'base64': base64,
                    'datetime': datetime,
                    'math': math
                }
                
                # Execute the code
                # Note: This is less secure than Docker but necessary if Docker is missing
                exec(code_to_exec, {"__builtins__": __builtins__}, local_vars)
                check_func = local_vars.get('check_log')
                
                if not check_func:
                    return {"error": "Function 'check_log(log_line: str) -> bool' not defined"}
                
                matched = []
                for line in logs_to_test:
                    try:
                        if check_func(line):
                            matched.append(line)
                    except:
                        continue
                
                return {"hits": len(matched), "matched_raws": matched}
                
        except Exception as e:
            return {"error": f"Local execution failed: {str(e)}"}
        finally:
            sys.stdin = old_stdin

    def test_pattern(
        self, code: str, store: str
    ) -> Tuple[bool, str, int]:
        """Test a Python script against every line in *store*."""
        if store not in self._stores:
            return False, f"Store '{store}' not found", 0
        
        results = self._run_in_docker(code, self._stores[store])
        
        if "error" in results:
            return False, results["error"], 0
        
        return True, "OK", results.get("hits", 0)

    def get_matches(
        self, code: str, store: str, limit: int = 10000
    ) -> List[Dict[str, Any]]:
        if store not in self._stores:
            return []
        
        results = self._run_in_docker(code, self._stores[store])
        if "error" in results:
            return []
            
        matched_raws = results.get("matched_raws", [])
        return [{"_raw": line} for line in matched_raws[:limit]]

    def get_matched_raws(
        self, code: str, store: str, limit: int = 10000
    ) -> Set[str]:
        if store not in self._stores:
            return set()
        
        results = self._run_in_docker(code, self._stores[store])
        if "error" in results:
            return set()
            
        matched_raws = results.get("matched_raws", [])
        return set(matched_raws[:limit])

    def count(self, store: str) -> int:
        return len(self._stores.get(store, []))

    def get_store_content(self, store: str) -> Set[str]:
        return set(self._stores.get(store, []))
