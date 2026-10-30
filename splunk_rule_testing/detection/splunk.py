import os
import re
import requests
import urllib3
import time
from typing import List, Dict, Tuple, Any, Optional

# TLS verification for the Splunk REST API.
#
# The published experiments ran against a local Splunk dev instance with a
# self-signed certificate, so verification is off by default and that default is
# preserved here for faithful replay. Set FORGE_SPLUNK_VERIFY_TLS=1 to verify.
# Nothing in the published results depends on this engine: the Table 3/5 numbers
# are produced by regex scoring (detection/regex.py) and never contact Splunk.
VERIFY_TLS = os.environ.get("FORGE_SPLUNK_VERIFY_TLS", "0") == "1"

if not VERIFY_TLS:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Splunk REST API endpoints
SPLUNK_JOBS_ENDPOINT = "/services/search/jobs"
SPLUNK_RECEIVER_ENDPOINT = "/services/receivers/simple"
SPLUNK_INDEX_ENDPOINT = "/services/data/indexes"


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class SplunkEngine:
    """Splunk-based detection proxy using REST API.

    Mirrors the Engine interface used by RegexEngine and PythonEngine.
    """

    def __init__(self, splunk_url: str, user: str, pwd: str) -> None:
        self.splunk_url = splunk_url.rstrip('/')
        self.user = user
        self.pwd = pwd

    # ------------------------------------------------------------------
    # Store (Index) management
    # ------------------------------------------------------------------

    def create_store(self, name: str) -> None:
        """Create (or reset) a Splunk index."""
        endpoint = f"{self.splunk_url}{SPLUNK_INDEX_ENDPOINT}"
        payload = {"name": name}
        try:
            resp = requests.post(endpoint, data=payload, auth=(self.user, self.pwd), verify=VERIFY_TLS, timeout=5)
            if resp.status_code == 201:
                print(f"{Colors.OKGREEN}[INFO] Created index '{name}'{Colors.ENDC}")
            elif resp.status_code == 409:
                pass
            else:
                print(f"{Colors.WARNING}[WARN] Could not create index '{name}': {resp.text[:100]}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}[WARN] Error creating index: {e}{Colors.ENDC}")

    def delete_store(self, name: str) -> None:
        """Delete a Splunk index if it exists."""
        endpoint = f"{self.splunk_url}{SPLUNK_INDEX_ENDPOINT}/{name}"
        try:
            resp = requests.delete(endpoint, auth=(self.user, self.pwd), verify=VERIFY_TLS, timeout=5)
            if resp.status_code == 200:
                print(f"{Colors.OKGREEN}[INFO] Deleted index '{name}'{Colors.ENDC}")
            else:
                pass
        except Exception as e:
            print(f"{Colors.WARNING}[WARN] Error deleting index: {e}{Colors.ENDC}")

    def list_stores(self) -> List[str]:
        """List all index names from Splunk."""
        endpoint = f"{self.splunk_url}{SPLUNK_INDEX_ENDPOINT}?output_mode=json&count=0"
        try:
            resp = requests.get(endpoint, auth=(self.user, self.pwd), verify=VERIFY_TLS, timeout=10)
            if resp.status_code == 200:
                return [e['name'] for e in resp.json().get('entry', [])]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_line(self, line: str, store: str, sourcetype: str = "manual", host: str = "harness") -> bool:
        """Add a single log line to a Splunk index."""
        endpoint = f"{self.splunk_url}{SPLUNK_RECEIVER_ENDPOINT}?index={store}&sourcetype={sourcetype}&host={host}"
        try:
            resp = requests.post(endpoint, data=line.encode(), auth=(self.user, self.pwd), verify=VERIFY_TLS, timeout=30)
            if resp.status_code < 300:
                return True
            print(f"{Colors.FAIL}Ingestion failed ({resp.status_code}) for index {store}: {resp.text}{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.FAIL}Ingestion Error for index {store}: {e}{Colors.ENDC}")
            return False

    def ingest_lines(self, lines: List[str], store: str, sourcetype: str = "manual", host: str = "harness", batch_size: int = 100) -> int:
        """Add multiple log lines to a Splunk index in batches. Returns the number of batches successfully sent."""
        non_empty = [l for l in lines if l and l.strip()]
        sent = 0
        for i in range(0, len(non_empty), batch_size):
            batch = non_empty[i:i + batch_size]
            combined = "\n".join(batch)
            if self.ingest_line(combined, store, sourcetype, host):
                sent += 1
        return sent

    # ------------------------------------------------------------------
    # Pattern matching (SPL)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_quotes(spl: str) -> str:
        """Replace single-quoted values on known top-level SPL field filters with double-quoted ones.
        Splunk silently ignores e.g. index='foo', returning 0 results.
        Only targets known top-level filter keywords to avoid corrupting single quotes
        inside rex/eval string literals such as \"Name='CommandLine'\"."""
        return re.sub(r"\b(index|sourcetype|source|host|splunk_server)='([^']*)'", r'\1="\2"', spl)

    def test_pattern(self, spl_query: str, store: Optional[str] = None, earliest: str = "0") -> Tuple[bool, str, int]:
        """Test an SPL query and return hit count. Note: 'store' is usually embedded in SPL as 'index=...'."""
        endpoint = f"{self.splunk_url}{SPLUNK_JOBS_ENDPOINT}"

        final_query = self._normalize_quotes(spl_query.strip())
        if not final_query.startswith("|") and not final_query.startswith("search"):
            final_query = f"search {final_query}"

        payload = {
            "search": final_query,
            "exec_mode": "blocking",
            "output_mode": "json",
            "earliest_time": earliest
        }

        try:
            response = requests.post(endpoint, data=payload, auth=(self.user, self.pwd), verify=VERIFY_TLS, timeout=20)
            if response.status_code >= 400:
                try:
                    msgs = response.json().get("messages", [])
                    error_text = "; ".join([m.get("text", "Unknown Error") for m in msgs])
                except Exception:
                    error_text = response.text[:200]
                return False, f"Splunk API {response.status_code}: {error_text}", 0

            sid = response.json().get("sid")
            if not sid:
                return False, "No SID returned", 0

            status_url = f"{self.splunk_url}/services/search/jobs/{sid}"
            status_response = requests.get(status_url, params={"output_mode": "json"}, auth=(self.user, self.pwd), verify=VERIFY_TLS, timeout=10)

            if status_response.status_code >= 400:
                return False, f"Status Error: {status_response.text}", 0

            entry = status_response.json().get("entry", [{}])[0]
            event_count = int(entry.get("content", {}).get("eventCount", 0))
            return True, "OK", event_count

        except Exception as e:
            return False, f"Connection Error: {str(e)}", 0

    def get_matches(self, spl_query: str, store: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetches full results for an SPL query."""
        endpoint = f"{self.splunk_url}{SPLUNK_JOBS_ENDPOINT}"

        final_query = self._normalize_quotes(spl_query.strip())
        if not final_query.startswith("|") and not final_query.startswith("search"):
            final_query = f"search {final_query}"

        payload = {
            "search": final_query,
            "exec_mode": "blocking",
            "output_mode": "json"
        }

        try:
            response = requests.post(endpoint, data=payload, auth=(self.user, self.pwd), verify=VERIFY_TLS, timeout=20)
            sid = response.json().get("sid")
            if not sid:
                return []

            results_url = f"{self.splunk_url}/services/search/jobs/{sid}/results"
            results_response = requests.get(results_url, params={"output_mode": "json", "count": limit}, auth=(self.user, self.pwd), verify=VERIFY_TLS, timeout=10)

            if results_response.status_code >= 400:
                return []

            return results_response.json().get("results", [])

        except Exception:
            return []

    def get_export_results(self, spl_query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Uses the export endpoint to fetch raw results. This is often more robust for large datasets."""
        endpoint = f"{self.splunk_url}/services/search/jobs/export"
        
        final_query = spl_query.strip()
        if not final_query.startswith("|") and not final_query.startswith("search"):
            final_query = f"search {final_query}"

        payload = {
            "search": final_query,
            "output_mode": "json",
            "earliest_time": "0"
        }
        
        results = []
        try:
            resp = requests.post(endpoint, data=payload, auth=(self.user, self.pwd), verify=VERIFY_TLS, stream=True, timeout=60)
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    results.append(data.get("result", data))
                    if limit and len(results) >= limit:
                        break
                except Exception:
                    continue
            return results
        except Exception:
            return []
