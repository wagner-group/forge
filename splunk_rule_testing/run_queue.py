#!/usr/bin/env python3
"""Sequential run queue for eval sweeps: hit run once, walk away.

Reads a JSON manifest of runs, executes them one at a time, and persists state
so an interrupted queue resumes exactly where it stopped. Built for the things
the eval sweeps actually hit:

* **Rate limits.** A run that dies on a 429 / quota / usage-limit message is not
  a failure -- the queue waits and retries the same run. Wait length comes from
  the Codex CLI's own rate-limit telemetry when available (sleep until the
  reported window reset), and falls back to doubling backoff when it is not.
  Rate-limit waits are capped by --max-rate-limit-waits and do NOT burn
  --max-attempts, which is reserved for real failures.
* **Proactive quota guard.** Before starting a run the queue reads the latest
  rate-limit snapshot the Codex CLI persisted and refuses to start when the
  remaining headroom in either window is below --min-headroom-pct. It then
  sleeps until that window resets rather than starting a doomed run.
* **Hard budget ceiling.** --budget-tokens / --max-runs halt the queue cleanly.
  Token spend is read from each run's <output-dir>/summary.json, so the ceiling
  works for the API-key arm, where there is no rate-limit telemetry at all.
* **Interruption.** Ctrl-C marks the in-flight run `interrupted` and exits
  cleanly. Re-running with the same state file picks it back up.

Runs that own an --output-dir are resumed via the harness's own --resume flag on
retry, so completed per-file reports are not regenerated.

Where the rate-limit numbers come from
--------------------------------------
`codex exec --json` does NOT print rate limits on stdout -- its event stream is
only thread.started / turn.started / item.completed / turn.completed, and only
turn.completed carries a `usage` object. The rate-limit snapshot lives in the
*persisted session rollout*, $CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl, as

    {"type": "event_msg", "payload": {"type": "token_count",
      "info": {...},
      "rate_limits": {"limit_id": "codex", "plan_type": "plus",
        "primary":   {"used_percent": 0.0, "window_minutes": 300,   "resets_at": <epoch>},
        "secondary": {"used_percent": 0.0, "window_minutes": 10080, "resets_at": <epoch>},
        "credits": {"has_credits": false, "unlimited": false, "balance": "0"},
        "spend_control_reached": null, "rate_limit_reached_type": null}}}

So the guard reads the newest rollout file. This costs nothing -- no API call,
no tokens. It only reflects ChatGPT-subscription auth (`codex login status` ->
"Logged in using ChatGPT"); on API-key billing no such snapshot is produced and
the guard degrades to the token budget.

Usage:
    python run_queue.py --manifest runs.json
    python run_queue.py --manifest runs.json --status
    python run_queue.py --manifest runs.json --retry-failed
    python run_queue.py --check-limits
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Phrases that mean "slow down", not "this run is broken".
#
# These are matched against child stdout, which carries dataset-derived log text
# and generated regexes, so bare substrings are unsafe: "429" matched inside
# "processed 1429 lines", and "capacity" appears in ordinary prose. Each entry
# must be a phrase specific to an error, and numeric codes are matched as
# standalone HTTP statuses by RATE_LIMIT_PATTERNS instead.
RATE_LIMIT_MARKERS = (
    "rate limit", "rate_limit", "ratelimit", "insufficient_quota",
    "quota exceeded", "exceeded your quota", "out of quota",
    "usage limit", "usage_limit", "too many requests",
    "please try again later", "retry after", "retry-after",
    "server overloaded", "over capacity", "at capacity",
)

# Standalone HTTP status shapes, so "1429" and "4290" do not match.
RATE_LIMIT_PATTERNS = (
    re.compile(r"(?<!\d)429(?!\d)\s*(?:[-:]|\b)"),
    re.compile(r"\bstatus(?:_code)?[\"'\s:=]+429(?!\d)", re.I),
    re.compile(r"\bhttp[/ ]?1\.[01]\s+429(?!\d)", re.I),
)

# Exit code a child may use to declare "I was rate limited" unambiguously.
RATE_LIMIT_EXIT_CODE = 3

# `used_percent` is reported in whole percentage points, so no cost below this
# is observable. A 387k-token batch did not move it at all.
PCT_QUANTISATION = 1.0

TERMINAL_OK = "done"
TERMINAL_FAIL = "failed"
HALTED = "halted"

STATE_VERSION = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_rate_limited(text: str, returncode: Optional[int] = None) -> bool:
    """Was this a rate limit rather than a broken run?

    A child that knows it was rate limited says so with RATE_LIMIT_EXIT_CODE;
    that is authoritative and checked first. Text sniffing is the fallback for
    children that do not, and is deliberately conservative because the text
    being searched contains dataset logs and generated regexes.
    """
    if returncode is not None and returncode == RATE_LIMIT_EXIT_CODE:
        return True
    low = text.lower()
    if any(m in low for m in RATE_LIMIT_MARKERS):
        return True
    return any(p.search(text) for p in RATE_LIMIT_PATTERNS)


def _fmt_epoch(epoch: Optional[int]) -> str:
    if not isinstance(epoch, (int, float)):
        return "?"
    return datetime.fromtimestamp(int(epoch), timezone.utc).isoformat(timespec="seconds")


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Codex rate-limit telemetry (read from persisted session rollouts; free)
# ---------------------------------------------------------------------------

def codex_home() -> str:
    return os.path.expanduser(os.environ.get("CODEX_HOME") or "~/.codex")


def _dig_rate_limits(obj: Any) -> Optional[Dict[str, Any]]:
    """Find a `rate_limits` dict anywhere in an event, tolerating reshuffles."""
    if isinstance(obj, dict):
        rl = obj.get("rate_limits")
        if isinstance(rl, dict):
            return rl
        for v in obj.values():
            found = _dig_rate_limits(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _dig_rate_limits(v)
            if found:
                return found
    return None


def _last_rate_limits_in_file(path: str) -> Optional[Dict[str, Any]]:
    """Last rate_limits payload in one rollout file (they are appended in order)."""
    found: Optional[Dict[str, Any]] = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"rate_limits"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rl = _dig_rate_limits(obj)
                if rl:
                    found = rl
    except OSError:
        return None
    return found


def read_rate_limit_snapshot(max_files: int = 5) -> Optional[Dict[str, Any]]:
    """Newest rate-limit snapshot the Codex CLI wrote, or None.

    Scans the most recently modified session rollouts. Returns the raw
    `rate_limits` payload plus `_source` / `_observed_epoch` / `_observed`.
    """
    base = os.path.join(codex_home(), "sessions")
    if not os.path.isdir(base):
        return None
    files = glob.glob(os.path.join(base, "**", "rollout-*.jsonl"), recursive=True)
    if not files:
        return None
    try:
        files.sort(key=os.path.getmtime, reverse=True)
    except OSError:
        pass
    for path in files[:max_files]:
        rl = _last_rate_limits_in_file(path)
        if rl:
            snap = dict(rl)
            try:
                mtime = int(os.path.getmtime(path))
            except OSError:
                mtime = int(time.time())
            snap["_source"] = path
            snap["_observed_epoch"] = mtime
            snap["_observed"] = _fmt_epoch(mtime)
            return snap
    return None


def window_headroom(snapshot: Dict[str, Any],
                    at_epoch: Optional[int] = None) -> List[Dict[str, Any]]:
    """Per-window remaining headroom, tightest first.

    A window whose `resets_at` is already in the past has rolled over since the
    snapshot was taken, so its `used_percent` is stale -- treat it as full.
    """
    at = int(at_epoch if at_epoch is not None else time.time())
    out: List[Dict[str, Any]] = []
    for name in ("primary", "secondary"):
        w = snapshot.get(name)
        if not isinstance(w, dict):
            continue
        used = w.get("used_percent")
        if not isinstance(used, (int, float)):
            continue
        resets_at = w.get("resets_at")
        rolled = isinstance(resets_at, (int, float)) and int(resets_at) <= at
        out.append({
            "window": name,
            "used_percent": float(used),
            "headroom_pct": 100.0 if rolled else max(0.0, 100.0 - float(used)),
            "window_minutes": w.get("window_minutes"),
            "resets_at": int(resets_at) if isinstance(resets_at, (int, float)) else None,
            "rolled_over": rolled,
        })
    out.sort(key=lambda d: d["headroom_pct"])
    return out


def snapshot_summary(snapshot: Optional[Dict[str, Any]]) -> str:
    if not snapshot:
        return "no Codex rate-limit telemetry found"
    parts = []
    for w in window_headroom(snapshot):
        mins = w.get("window_minutes")
        label = f"{w['window']}({mins}m)" if mins else w["window"]
        tag = " [rolled over]" if w["rolled_over"] else ""
        parts.append(f"{label} {w['headroom_pct']:.1f}% left, "
                     f"resets {_fmt_epoch(w['resets_at'])}{tag}")
    plan = snapshot.get("plan_type") or "?"
    head = f"plan={plan} observed={snapshot.get('_observed', '?')}"
    if not parts:
        return f"{head} (no windows reported)"
    return head + " | " + "; ".join(parts)


def hard_stop_reason(snapshot: Optional[Dict[str, Any]]) -> Optional[str]:
    """Conditions that mean 'do not start anything', independent of headroom.

    Note: `credits.has_credits == false` is NOT one of these. On a Plus plan
    that just means no pay-as-you-go credits were purchased; the plan's own
    windows are what gate usage.
    """
    if not snapshot:
        return None
    if snapshot.get("spend_control_reached"):
        return "spend_control_reached is set in the Codex rate-limit snapshot"
    return None


def reset_wait_seconds(snapshot: Optional[Dict[str, Any]], fallback: int,
                       *, only_exhausted: bool = True,
                       exhausted_at_pct: float = 95.0,
                       blocking_window: Optional[str] = None) -> Tuple[int, str]:
    """How long to sleep for a window to reset. Returns (seconds, reason).

    `blocking_window` names the window that is actually holding the queue up. It
    matters: taking min() over every future reset means a soon-resetting 5-hour
    window is chosen while the 7-day window is the one blocking, so the queue
    wakes, finds itself still blocked, and sleeps the same short interval forever.
    """
    if not snapshot:
        return fallback, f"no telemetry; falling back to {fallback}s"
    at = int(time.time())
    cands: List[Tuple[int, str]] = []
    for w in window_headroom(snapshot, at):
        if w["resets_at"] is None or w["resets_at"] <= at:
            continue
        if blocking_window and w["window"] != blocking_window:
            continue
        if only_exhausted and not blocking_window and w["used_percent"] < exhausted_at_pct:
            continue
        cands.append((w["resets_at"], w["window"]))
    if not cands and blocking_window:
        # The blocking window has no future reset in this snapshot; do not fall
        # back to a different window's shorter reset.
        return fallback, (f"{blocking_window} window is blocking but has no future "
                          f"reset in the snapshot; falling back to {fallback}s")
    if not cands and only_exhausted:
        return reset_wait_seconds(snapshot, fallback, only_exhausted=False)
    if not cands:
        return fallback, f"no future reset in snapshot; falling back to {fallback}s"
    resets_at, name = min(cands)
    # +30s so we wake up after the window has actually rolled over.
    return max(60, resets_at - at + 30), f"{name} window resets at {_fmt_epoch(resets_at)}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class QueueState:
    """State file. Backward compatible: v1 files ({"updated","runs"}) load fine
    and gain a `meta` block on the next save."""

    def __init__(self, path: str, cost_path: Optional[str] = None) -> None:
        self.path = path
        # Cost history lives in its own file so what the queue learns about
        # per-run window cost survives across manifests and state files. A fresh
        # sweep should not start blind just because it has a new state file.
        self.cost_path = cost_path
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.meta: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    blob = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise SystemExit(f"[queue] state file {path} is unreadable: {e}")
            if isinstance(blob, dict):
                self.runs = blob.get("runs") or {}
                self.meta = blob.get("meta") or {}
        self.meta.setdefault("tokens_spent", 0)
        self.meta.setdefault("runs_with_token_data", 0)
        self._load_shared_cost_history()

    def _load_shared_cost_history(self) -> None:
        if not self.cost_path or not os.path.exists(self.cost_path):
            return
        try:
            with open(self.cost_path, encoding="utf-8") as f:
                blob = json.load(f)
        except (json.JSONDecodeError, OSError):
            return  # corrupt shared history must never block a sweep
        shared = (blob or {}).get("window_cost_history") or {}
        # REPLACE, do not concatenate. The shared file is the single source of
        # truth. Merging it onto history already loaded from the state file, then
        # saving the merged list back, doubles the history on every invocation
        # (1 -> 3 -> 7 -> 14 entries for three real runs) and destroys recency
        # ordering, which --cost-window depends on.
        hist = {}
        for w, vals in shared.items():
            if isinstance(vals, list):
                hist[w] = [float(v) for v in vals
                           if isinstance(v, (int, float))][-200:]
        self.meta["window_cost_history"] = hist

    def _save_shared_cost_history(self) -> None:
        if not self.cost_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cost_path) or ".", exist_ok=True)
            tmp = self.cost_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"updated": now(),
                           "window_cost_history": self.meta.get("window_cost_history", {})},
                          f, indent=2)
            os.replace(tmp, self.cost_path)
        except OSError as e:
            print(f"[queue] warning: could not write cost history {self.cost_path}: {e}")

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"updated": now(), "version": STATE_VERSION,
                       "meta": self.meta, "runs": self.runs}, f, indent=2)
        os.replace(tmp, self.path)

    def get(self, name: str) -> Dict[str, Any]:
        return self.runs.setdefault(name, {"status": "queued", "attempts": 0})

    def set(self, name: str, **kw: Any) -> None:
        self.get(name).update(kw)
        self.save()

    # -- budget ------------------------------------------------------------
    def record_tokens(self, name: str, tokens: Optional[int]) -> None:
        if not tokens:
            return
        self.get(name)["tokens"] = tokens
        self.meta["tokens_spent"] = int(self.meta.get("tokens_spent", 0)) + int(tokens)
        self.meta["runs_with_token_data"] = int(self.meta.get("runs_with_token_data", 0)) + 1
        self.save()

    def mean_tokens_per_run(self) -> int:
        n = int(self.meta.get("runs_with_token_data", 0))
        if n <= 0:
            return 0
        return int(int(self.meta.get("tokens_spent", 0)) / n)

    # -- observed cost per run, in window-percent ---------------------------
    def record_window_cost(self, name: str,
                           before: Optional[Dict[str, Any]],
                           after: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Record how much of each rolling window one run actually consumed.

        Percent is the only currency the rate-limit API speaks, so cost per run
        is learned by differencing used_percent across the run. Two cases are
        discarded rather than recorded:

          * the window rolled over mid-run (delta goes negative), and
          * either snapshot is missing (API-key billing, or codex never ran).

        The delta is an UPPER BOUND on this run's own cost: the window is shared
        with any other Codex use on the account, so concurrent interactive
        sessions inflate it. Erring high is the safe direction for a guard.
        """
        deltas: Dict[str, float] = {}
        if not before or not after:
            return deltas
        # If the rollout did not advance during the run, before and after are the
        # same payload and every delta is a spurious, confident-looking 0.0.
        if before == after:
            return deltas
        b_obs, a_obs = before.get("_observed_epoch"), after.get("_observed_epoch")
        if isinstance(b_obs, (int, float)) and isinstance(a_obs, (int, float)) \
                and a_obs <= b_obs:
            return deltas
        for w in ("primary", "secondary"):
            b, a = before.get(w), after.get(w)
            if not isinstance(b, dict) or not isinstance(a, dict):
                continue
            bu, au = b.get("used_percent"), a.get("used_percent")
            if not isinstance(bu, (int, float)) or not isinstance(au, (int, float)):
                continue
            if b.get("resets_at") != a.get("resets_at"):
                continue  # window rolled over; delta is meaningless
            d = float(au) - float(bu)
            if d < 0:
                continue
            deltas[w] = d
        if deltas:
            self.get(name)["window_cost_pct"] = deltas
            hist = self.meta.setdefault("window_cost_history", {})
            for w, d in deltas.items():
                hist.setdefault(w, []).append(round(d, 4))
            self.save()
            self._save_shared_cost_history()
        return deltas

    def predicted_pct_per_run(self, window: str, recent: int = 5) -> Optional[float]:
        """Worst cost among the most RECENT `recent` runs, or None with no history.

        Max, not mean: a guard should size against the expensive run, and per-run
        cost varies by an order of magnitude with pool size.

        But max over a *sliding window*, not over all time -- otherwise one costly
        run early on pins the estimate high forever and the queue never recovers
        its throughput. Bounding to recent history lets the estimate come back
        down as cheaper runs accumulate.
        """
        hist = (self.meta.get("window_cost_history") or {}).get(window) or []
        vals = [float(x) for x in hist if isinstance(x, (int, float)) and x >= 0]
        if not vals:
            return None
        est = max(vals[-recent:] if recent and recent > 0 else vals)
        # `used_percent` is quantised to whole points, so anything below one step
        # is indistinguishable from "this run cost nothing measurable". Returning
        # 0.0 here would read downstream as a confident estimate of zero and make
        # the predictive check silently always-pass. Report no signal instead.
        if est < PCT_QUANTISATION:
            return None
        return est

    def observed_runs(self, window: str) -> int:
        return len((self.meta.get("window_cost_history") or {}).get(window) or [])

    def record_snapshot(self, snapshot: Optional[Dict[str, Any]]) -> None:
        if snapshot:
            self.meta["rate_limits"] = snapshot
            self.meta["rate_limits_seen"] = now()
            self.save()


# ---------------------------------------------------------------------------
# Token accounting for a finished run
# ---------------------------------------------------------------------------

def entry_output_dir(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("output_dir"):
        return entry["output_dir"]
    args = list(entry.get("args", []))
    for i, a in enumerate(args):
        if a == "--output-dir" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--output-dir="):
            return a.split("=", 1)[1]
    return None


def read_run_tokens(entry: Dict[str, Any]) -> Optional[int]:
    """Total tokens for a finished run, from <output-dir>/summary.json."""
    out = entry_output_dir(entry)
    if not out:
        return None
    path = os.path.join(out, "summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    tt = summary.get("total_tokens")
    if isinstance(tt, dict):
        return sum(v for v in tt.values() if isinstance(v, int))
    if isinstance(tt, int):
        return tt
    return None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

_stop = {"requested": False}


def _handle_sigint(signum, frame):  # noqa: ARG001
    if _stop["requested"]:
        print("\n[queue] second interrupt - exiting immediately.")
        sys.exit(130)
    _stop["requested"] = True
    print("\n[queue] interrupt received - finishing current run's cleanup, then stopping.")
    raise KeyboardInterrupt


def interruptible_sleep(seconds: float, *, label: str = "", tick: int = 10) -> None:
    """Sleep against a wall-clock deadline, checking for Ctrl-C."""
    deadline = time.time() + max(0.0, seconds)
    announced = False
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        if _stop["requested"]:
            raise KeyboardInterrupt
        if label and not announced:
            print(f"[queue] waiting {_fmt_dur(remaining)} - {label}")
            announced = True
        time.sleep(min(tick, remaining))


class Guard:
    """Proactive quota / budget guard. Consulted before every run."""

    def __init__(self, *, min_headroom_pct: float, budget_tokens: int,
                 max_runs: int, max_wait: int, ignore_limits: bool,
                 safety_factor: float = 1.25, cost_window: int = 5,
                 parallel: int = 1) -> None:
        self.safety_factor = safety_factor
        self.cost_window = cost_window
        # How many runs may be in flight at once. The executor is sequential
        # today, so this is 1; the estimate is written in terms of it so that
        # wave sizing is correct the moment a parallel executor lands.
        self.parallel = max(1, parallel)
        # Which window most recently caused a "wait", so the sleep is taken from
        # that window's reset rather than whichever resets soonest.
        self.blocking_window: Optional[str] = None
        self.min_headroom_pct = min_headroom_pct
        self.budget_tokens = budget_tokens
        self.max_runs = max_runs
        self.max_wait = max_wait
        self.ignore_limits = ignore_limits
        self.runs_started = 0

    def refresh(self, state: QueueState) -> Optional[Dict[str, Any]]:
        snap = read_rate_limit_snapshot()
        state.record_snapshot(snap)
        return snap

    def check_budget(self, state: QueueState) -> Optional[str]:
        """Return a halt reason, or None to proceed."""
        if self.max_runs and self.runs_started >= self.max_runs:
            return f"--max-runs {self.max_runs} reached ({self.runs_started} started)"
        if self.budget_tokens:
            spent = int(state.meta.get("tokens_spent", 0))
            est = state.mean_tokens_per_run()
            if spent >= self.budget_tokens:
                return (f"--budget-tokens {self.budget_tokens:,} exhausted "
                        f"({spent:,} spent)")
            if est and spent + est > self.budget_tokens:
                return (f"--budget-tokens {self.budget_tokens:,} would be exceeded: "
                        f"{spent:,} spent + {est:,} est/run > {self.budget_tokens:,}")
        return None

    def check_headroom(self, state: QueueState) -> Tuple[str, str]:
        """Return (action, detail). action in {'go', 'wait', 'halt'}."""
        self.blocking_window = None
        if self.ignore_limits:
            return "go", "--ignore-rate-limits set"
        snap = self.refresh(state)
        if not snap:
            return "go", ("no Codex rate-limit telemetry (API-key billing, or codex "
                          "has never run here) - relying on the token budget only")
        stop = hard_stop_reason(snap)
        if stop:
            return "halt", stop
        windows = window_headroom(snap)
        if not windows:
            return "go", "snapshot has no usable windows"
        tightest = windows[0]
        if tightest["headroom_pct"] < self.min_headroom_pct:
            self.blocking_window = tightest["window"]
            return "wait", (f"{tightest['window']} window at "
                            f"{tightest['used_percent']:.1f}% used "
                            f"({tightest['headroom_pct']:.1f}% left, "
                            f"below --min-headroom-pct {self.min_headroom_pct})")

        # Predictive check: will the NEXT run actually fit in what is left?
        # The static floor above only asks whether some headroom remains, which
        # lets a run start on 12% when the last one cost 40%.
        for w in windows:
            est = state.predicted_pct_per_run(w["window"], self.cost_window)
            if est is None:
                continue
            need = est * self.safety_factor * self.parallel
            if need > w["headroom_pct"]:
                self.blocking_window = w["window"]
                return "wait", (
                    f"{w['window']} window has {w['headroom_pct']:.1f}% left but the "
                    f"costliest of {state.observed_runs(w['window'])} prior run(s) used "
                    f"{est:.1f}% (x{self.safety_factor:g} safety = {need:.1f}% needed)")

        est_p = state.predicted_pct_per_run(tightest["window"], self.cost_window)
        n_obs = state.observed_runs(tightest["window"])
        if est_p is not None:
            est_txt = f", est {est_p:.1f}%/run from {n_obs} run(s)"
        elif n_obs:
            # Observations exist but every one is below the 1-point quantisation.
            est_txt = (f", {n_obs} run(s) observed but each cost <{PCT_QUANTISATION:g}% "
                       f"(below telemetry resolution) - no usable estimate")
        else:
            est_txt = ", no cost history yet"
        return "go", (f"{tightest['window']} window {tightest['headroom_pct']:.1f}% left{est_txt}")

    def await_headroom(self, state: QueueState, name: str) -> Optional[str]:
        """Block until there is headroom. Returns a halt reason, or None.

        Total time spent waiting is accumulated and checked against --max-wait.
        Comparing only a single sleep against it lets the queue loop forever:
        if the 7-day window is blocking while the 5-hour window resets in
        minutes, every individual sleep passes the check and none of them help.
        """
        waited = 0
        while True:
            action, detail = self.check_headroom(state)
            if action == "go":
                print(f"[queue] limits ok for {name}: {detail}")
                return None
            if action == "halt":
                return detail
            snap = state.meta.get("rate_limits")
            wait, reason = reset_wait_seconds(snap, 900, only_exhausted=False,
                                              blocking_window=self.blocking_window)
            if self.max_wait and waited + wait > self.max_wait:
                return (f"{detail}; would exceed --max-wait "
                        f"{_fmt_dur(self.max_wait)} (already waited "
                        f"{_fmt_dur(waited)}, next reset {_fmt_dur(wait)} away)")
            state.set(name, status="waiting_quota", note=detail)
            print(f"[queue] HOLDING {name}: {detail} "
                  f"(waited {_fmt_dur(waited)} so far)")
            interruptible_sleep(wait, label=reason)
            waited += wait


def run_one(entry: Dict[str, Any], state: QueueState, *,
            python_bin: str, base_script: str, log_dir: str,
            max_attempts: int, base_backoff: int, dry_run: bool,
            max_rate_limit_waits: int = 8,
            guard: Optional[Guard] = None) -> str:
    """Execute one manifest entry to a terminal status."""
    name = entry["name"]
    st = state.get(name)
    backoff = base_backoff
    consecutive_rl = 0
    # Rate-limit retries are tracked separately so they do not burn the
    # --max-attempts budget, which is reserved for real failures.
    rl_waits = 0
    failures = 0

    while failures < max_attempts:
        attempt = st.get("attempts", 0) + 1
        argv = [python_bin, base_script] + list(entry.get("args", []))

        # On a retry, let the harness skip work it already finished.
        #
        # Gate on whether output already exists, NOT on attempt number:
        # --retry-failed resets attempts to 0, so an attempt-based gate silently
        # dropped --resume in exactly the expensive case it exists for, redoing a
        # whole completed batch.
        out_dir = entry_output_dir(entry)
        has_prior_output = bool(out_dir) and os.path.isdir(out_dir) and bool(os.listdir(out_dir))
        if (attempt > 1 or has_prior_output) and entry.get("resumable", True) \
                and "--resume" not in argv:
            argv.append("--resume")

        env = dict(os.environ)
        env.update({k: str(v) for k, v in (entry.get("env") or {}).items()})

        print(f"\n[queue] === {name} (attempt {attempt}, failures {failures}/{max_attempts}) ===")
        print(f"[queue] {' '.join(argv)}")
        if dry_run:
            state.set(name, status=TERMINAL_OK, attempts=attempt, note="dry-run")
            return TERMINAL_OK

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.attempt{attempt}.log")
        state.set(name, status="running", attempts=attempt,
                  started=now(), log=log_path)
        # Snapshot before the run so its window cost can be differenced after.
        snap_before = dict(state.meta.get("rate_limits") or {}) or None
        if guard is not None and attempt == 1:
            # Count runs, not attempts: incrementing per attempt made
            # --max-runs 10 halt after as few as 4 real runs.
            guard.runs_started += 1

        tail: List[str] = []
        proc: Optional[subprocess.Popen] = None
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, text=True, bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    logf.write(line)
                    logf.flush()
                    sys.stdout.write(line)
                    tail.append(line)
                    if len(tail) > 200:
                        tail.pop(0)
                rc = proc.wait()
        except KeyboardInterrupt:
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=30)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            state.set(name, status="interrupted", ended=now())
            raise

        blob = "".join(tail)

        # Record spend once per run, not once per attempt. read_run_tokens reads
        # the CUMULATIVE <output-dir>/summary.json, so calling it after every
        # attempt counted a retried 100k run as 300k and made --budget-tokens
        # halt at a third of the real ceiling.
        tokens_now = read_run_tokens(entry)
        if guard is not None:
            snap_after = guard.refresh(state)
            deltas = state.record_window_cost(name, snap_before, snap_after)
            if deltas:
                print("[queue] window cost of this run: "
                      + ", ".join(f"{w} {d:.2f}%" for w, d in sorted(deltas.items())))

        if rc == 0:
            state.record_tokens(name, tokens_now)
            state.set(name, status=TERMINAL_OK, ended=now(), returncode=0)
            print(f"[queue] {name}: done")
            return TERMINAL_OK

        if is_rate_limited(blob, rc):
            consecutive_rl += 1
            rl_waits += 1
            if rl_waits > max_rate_limit_waits:
                state.record_tokens(name, tokens_now)
                state.set(name, status=TERMINAL_FAIL, ended=now(), returncode=rc,
                          note=f"gave up after {max_rate_limit_waits} rate-limit waits")
                print(f"[queue] {name}: FAILED - {max_rate_limit_waits} rate-limit "
                      f"waits exhausted")
                return TERMINAL_FAIL
            snap = state.meta.get("rate_limits")
            fallback = backoff * (2 ** (consecutive_rl - 1))
            wait, reason = reset_wait_seconds(
                snap, fallback,
                blocking_window=(guard.blocking_window if guard is not None else None))
            state.set(name, status="rate_limited", ended=now(),
                      returncode=rc, note=f"wait {wait}s ({reason})")
            print(f"[queue] {name}: rate limited (rc={rc}). "
                  f"Sleeping {_fmt_dur(wait)} - {reason}")
            interruptible_sleep(wait)
            continue

        consecutive_rl = 0
        failures += 1
        state.set(name, status="error", ended=now(), returncode=rc)
        print(f"[queue] {name}: exited {rc} (not a rate limit). "
              f"{max_attempts - failures} attempt(s) left.")

    state.record_tokens(name, read_run_tokens(entry))
    state.set(name, status=TERMINAL_FAIL, ended=now())
    print(f"[queue] {name}: FAILED after {max_attempts} attempts")
    return TERMINAL_FAIL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_status(manifest: List[Dict[str, Any]], state: QueueState) -> None:
    print(f"{'run':<34} {'status':<15} {'att':<4} {'tokens':>12}  {'log'}")
    print("-" * 100)
    for entry in manifest:
        st = state.get(entry["name"])
        tok = st.get("tokens")
        tok_s = f"{tok:,}" if isinstance(tok, int) else "-"
        print(f"{entry['name']:<34} {st.get('status','queued'):<15} "
              f"{st.get('attempts',0):<4} {tok_s:>12}  {st.get('log','')}")
    spent = int(state.meta.get("tokens_spent", 0))
    if spent:
        print(f"\n[budget] {spent:,} tokens recorded over "
              f"{state.meta.get('runs_with_token_data', 0)} run(s); "
              f"mean {state.mean_tokens_per_run():,}/run")
    snap = state.meta.get("rate_limits")
    if snap:
        print(f"[limits] {snapshot_summary(snap)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="JSON file describing the runs")
    ap.add_argument("--state", default=None, help="State file (default: <manifest>.state.json)")
    ap.add_argument("--log-dir", default="queue_logs")
    ap.add_argument("--python", default=sys.executable, help="Interpreter for child runs")
    ap.add_argument("--script", default="evaluate_test_split.py")
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="Real (non-rate-limit) failures tolerated per run")
    ap.add_argument("--max-rate-limit-waits", type=int, default=8,
                    help="Rate-limit waits tolerated per run before giving up")
    ap.add_argument("--backoff", type=int, default=900,
                    help="Fallback rate-limit backoff in seconds when telemetry is "
                         "unavailable (default 900 = 15m)")
    ap.add_argument("--min-headroom-pct", type=float, default=10.0,
                    help="Refuse to start a run when the tightest rate-limit window "
                         "has less than this %% left (default 10)")
    ap.add_argument("--max-wait", type=int, default=8 * 3600,
                    help="Give up instead of waiting longer than this for a window "
                         "reset (seconds, default 8h; 0 = wait forever)")
    ap.add_argument("--budget-tokens", type=int, default=0,
                    help="Halt the queue before starting a run that would push "
                         "cumulative tokens past this ceiling (0 = no ceiling)")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="Halt after starting this many runs in one invocation "
                         "(0 = no ceiling)")
    ap.add_argument("--ignore-rate-limits", action="store_true",
                    help="Skip the proactive headroom guard (reactive backoff stays)")
    ap.add_argument("--cost-history",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         ".run_queue_cost_history.json"),
                    help="Where observed per-run window cost is accumulated across "
                         "sweeps, so estimates are not relearned each time")
    ap.add_argument("--cost-window", type=int, default=5,
                    help="Size the estimate off the worst of the last N runs, so a "
                         "single expensive run does not pin it high forever (0 = all)")
    ap.add_argument("--safety-factor", type=float, default=1.25,
                    help="Multiply the observed worst per-run window cost by this "
                         "when deciding whether the next run fits (default 1.25)")
    ap.add_argument("--check-limits", action="store_true",
                    help="Print the current Codex rate-limit snapshot and exit")
    ap.add_argument("--status", action="store_true", help="Print queue status and exit")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Reset failed/error/interrupted runs back to queued")
    ap.add_argument("--only", nargs="*", help="Run only these entry names")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = ap.parse_args()

    if args.check_limits:
        snap = read_rate_limit_snapshot()
        print(snapshot_summary(snap))
        if snap:
            print(f"[source] {snap.get('_source')}")
            print(json.dumps({k: v for k, v in snap.items()
                              if not k.startswith("_")}, indent=2))
            return 0
        return 1

    if not args.manifest:
        ap.error("--manifest is required (except with --check-limits)")

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    if isinstance(manifest, dict):
        manifest = manifest.get("runs", [])

    state = QueueState(args.state or args.manifest + ".state.json",
                       cost_path=args.cost_history)

    if args.status:
        print_status(manifest, state)
        return 0

    if args.retry_failed:
        n = 0
        for entry in manifest:
            st = state.get(entry["name"])
            if st.get("status") in (TERMINAL_FAIL, "error", "interrupted",
                                    "rate_limited", "waiting_quota", HALTED):
                st.update({"status": "queued", "attempts": 0})
                n += 1
        state.save()
        print(f"[queue] reset {n} run(s) to queued")

    signal.signal(signal.SIGINT, _handle_sigint)

    guard = Guard(safety_factor=args.safety_factor,
                  cost_window=args.cost_window,
                  min_headroom_pct=args.min_headroom_pct,
                  budget_tokens=args.budget_tokens,
                  max_runs=args.max_runs,
                  max_wait=args.max_wait,
                  ignore_limits=args.ignore_rate_limits)

    pending = [e for e in manifest
               if state.get(e["name"]).get("status") != TERMINAL_OK
               and (not args.only or e["name"] in args.only)]

    print(f"[queue] {len(pending)} run(s) pending of {len(manifest)} total")
    if not args.dry_run:
        print(f"[limits] {snapshot_summary(guard.refresh(state))}")

    results: Dict[str, str] = {}
    halted: Optional[str] = None
    try:
        for entry in pending:
            name = entry["name"]
            if not args.dry_run:
                halted = guard.check_budget(state)
                if halted:
                    break
                halted = guard.await_headroom(state, name)
                if halted:
                    break
            results[name] = run_one(
                entry, state,
                python_bin=args.python, base_script=args.script,
                log_dir=args.log_dir, max_attempts=args.max_attempts,
                base_backoff=args.backoff, dry_run=args.dry_run,
                max_rate_limit_waits=args.max_rate_limit_waits,
                guard=guard,
            )
    except KeyboardInterrupt:
        print("\n[queue] stopped. Re-run the same command to resume.")
        print_status(manifest, state)
        return 130

    if halted:
        print(f"\n[queue] HALTED before starting more runs: {halted}")
        print("[queue] nothing was started past this point; re-run to continue.")
        print_status(manifest, state)
        return 2

    print("\n[queue] all pending runs finished.")
    print_status(manifest, state)
    return 0 if all(v == TERMINAL_OK for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
