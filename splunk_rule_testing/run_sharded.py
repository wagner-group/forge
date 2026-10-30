#!/usr/bin/env python3
"""Run one FORGE arm across N concurrent shards, then merge into one report.

Why this exists: a FORGE trial spends ~100% of its wall clock blocked on the
model API (measured: 0.0% CPU, ~31s per API call, ~127s per trial). So the work
is latency-bound, not compute-bound, and running N processes costs almost no
local resources while cutting wall clock close to linearly. A continuous arm
drops from ~65h to ~8h at 8 shards.

    python3 run_sharded.py --shards 8 \
      --output-dir reports_dir/forge_luna_cont_desc \
      -- --regex-mode --dir ../dataset/final --test-split 0.5 --num-trial 10 \
         --N 10 --seed 161 --provider openai --openai-model gpt-5.6-luna \
         --openai-effort none

Everything after `--` is passed through to evaluate_test_split.py, except
--dir and --output-dir, which this owns.

Shards are balanced by estimated wall clock (trials x calls-per-trial, calibrated
against a completed arm) rather than file count, so they finish together. Each shard retries with
exponential backoff, and rate limits are distinguished from real failures the
same way run_queue.py does it. Retries pass --resume, so completed files are
never redone.

Concurrency is adaptive: each rate-limit classification lowers the launch
ceiling by one for --rl-window seconds (default 600), stepping the fleet down
from --shards toward --min-shards (default 4). Events expire on their own, so
the ceiling climbs back to full width once the pressure stops. Running shards
are never killed to meet a lowered ceiling -- the ceiling gates launching only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the real aggregation and rate-limit classification rather than
# reimplementing either.
_ets = _load("_ets", HERE / "evaluate_test_split.py")
_rq = _load("_rq", HERE / "run_queue.py")


def estimate_cost(path: Path, max_seeds: int) -> float:
    """Per-file wall-clock estimate = trials x API calls per trial.

    Calibrated against a completed arm rather than guessed. Measured there:
    calls/trial has mean 3.97 and stdev 0.63, and correlates only weakly with
    pool size (r=+0.46, ranging 3.5 calls at pool=2 to 6.0 at pool=3691). Since
    each call costs roughly constant latency (~31s), wall clock tracks TRIALS,
    with pool size a mild logarithmic adjustment.

    An earlier version used trials x pool, which overweighted pool by orders of
    magnitude and put one big file alone on a shard estimated at 2.7x the load
    of nine files elsewhere.
    """
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 1.0
    tps = len(d.get("true_positives") or [])
    tns = len(d.get("true_negatives") or [])
    trials = max(1, min(tps, max_seeds))
    calls_per_trial = 3.3 + 0.75 * math.log10(max(tps + tns, 2))
    return trials * calls_per_trial


# Files observed to wedge the agent reproducibly: an ESTABLISHED connection that
# passes no data, at 0% CPU, cleared only by the stall watchdog. The watchdog
# relaunches, --resume restarts on the SAME file, and it wedges again -- so a
# wedging file blocks every file queued behind it on its shard indefinitely.
# Pinning each to its own shard caps the damage at that one file instead of
# stalling three or four others. Measured effect: no_desc runs were taking 42-62
# min against 19-37 for desc, almost entirely this tail.
PATHOLOGICAL = ("get_domainuser_with_powershell_script_block",)


def balance(files: List[Path], shards: int, max_seeds: int) -> List[List[Path]]:
    """Greedy longest-processing-time first, with known wedging files isolated."""
    pinned = [f for f in files if f.stem in PATHOLOGICAL]
    rest = [f for f in files if f.stem not in PATHOLOGICAL]

    # One solo shard per pathological file, taken off the top.
    solo = [[f] for f in pinned][:max(0, shards - 1)]
    remaining_shards = max(1, shards - len(solo))

    weighted = sorted(((estimate_cost(f, max_seeds), f) for f in rest),
                      key=lambda t: -t[0])
    buckets: List[List[Path]] = [[] for _ in range(remaining_shards)]
    loads = [0.0] * remaining_shards
    for cost, f in weighted:
        i = loads.index(min(loads))
        buckets[i].append(f)
        loads[i] += cost
    return solo + [b for b in buckets if b]


def seed_shard_outputs(buckets: List[List[Path]], work: Path,
                       seed_from: Optional[Path]) -> int:
    """Pre-populate shard output dirs from a previous partial run.

    Each shard passes --resume, which skips any file that already has a report in
    that shard's own output dir. Copying an interrupted run's per-file reports
    into the right shard means a resumed arm redoes only what is genuinely
    missing instead of starting over.
    """
    if not seed_from:
        return 0
    src = Path(seed_from) / "files"
    if not src.is_dir():
        print(f"  --seed-from: no files/ under {seed_from}; nothing to reuse")
        return 0
    owner = {}
    for i, files in enumerate(buckets):
        for f in files:
            owner[f.name] = i
    n = 0
    for rep in sorted(src.glob("*.json")):
        i = owner.get(rep.name)
        if i is None:
            continue
        dst = work / f"out_{i:02d}" / "files"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rep, dst / rep.name)
        n += 1
    return n


def make_shard_dirs(buckets: List[List[Path]], work: Path) -> List[Path]:
    """One input dir per shard, symlinking the dataset files (no copying)."""
    if work.exists():
        shutil.rmtree(work)
    dirs = []
    for i, files in enumerate(buckets):
        d = work / f"shard_{i:02d}"
        d.mkdir(parents=True)
        for f in files:
            (d / f.name).symlink_to(f.resolve())
        dirs.append(d)
    return dirs


class ConcurrencyGovernor:
    """Adaptive shard width: back off the number of *concurrent* shards when
    rate limits cluster, and recover automatically when they stop.

    Why not just per-shard backoff: each shard already retries with exponential
    backoff, but if the limit is account-wide then 16 shards each backing off
    independently still re-converge on the same wall and re-trip it. Narrowing
    the fleet is the only lever that reduces aggregate request rate.

    The rule: every rate-limit classification drops the launch ceiling by one
    for `window` seconds. Events expire on their own, so the ceiling climbs back
    to full width once the pressure stops -- there is no separate recovery
    state machine to get stuck in. The ceiling never falls below `floor`, so a
    persistent limiter degrades to a slow run rather than a stalled one.

    Running shards are never killed to meet a lowered ceiling; the ceiling only
    gates *launching*, so in-flight work always finishes.
    """

    def __init__(self, total: int, floor: int, window: float) -> None:
        self.total = max(1, total)
        self.floor = max(1, min(floor, self.total))
        self.window = max(0.0, window)
        self.events: List[float] = []
        self._last_limit = self.total

    def _prune(self, now: float) -> None:
        cut = now - self.window
        if self.events and self.events[0] <= cut:
            self.events = [t for t in self.events if t > cut]

    def record(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self._prune(now)
        self.events.append(now)

    def limit(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        self._prune(now)
        return max(self.floor, self.total - len(self.events))

    def can_launch(self, running: int, now: Optional[float] = None) -> bool:
        return running < self.limit(now)

    def poll_report(self, now: Optional[float] = None) -> Optional[str]:
        """One-line message when the ceiling moves, else None (keeps logs quiet)."""
        lim = self.limit(now)
        if lim == self._last_limit:
            return None
        direction = "narrowing" if lim < self._last_limit else "widening"
        msg = (f"[width] {direction} concurrency {self._last_limit} -> {lim} "
               f"(of {self.total}, floor {self.floor}; "
               f"{len(self.events)} rate-limit event(s) in last {self.window:.0f}s)")
        self._last_limit = lim
        return msg


class Shard:
    def __init__(self, idx: int, in_dir: Path, out_dir: Path, passthrough: List[str],
                 python_bin: str, script: Path) -> None:
        self.idx = idx
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.passthrough = passthrough
        self.python_bin = python_bin
        self.script = script
        self.attempts = 0
        self.rl_waits = 0
        self.proc: Optional[subprocess.Popen] = None
        self.log = out_dir.parent / f"_shard_{idx:02d}.log"
        self.done = False
        self.failed = False
        self.wake_at = 0.0
        self.stalls = 0

    def n_files(self) -> int:
        return len(list(self.in_dir.glob("*.json")))

    def n_done(self) -> int:
        fd = self.out_dir / "files"
        return len(list(fd.glob("*.json"))) if fd.is_dir() else 0

    def launch(self) -> None:
        self.attempts += 1
        argv = [self.python_bin, str(self.script), *self.passthrough,
                "--dir", str(self.in_dir), "--output-dir", str(self.out_dir)]
        # Always resume: on a first launch the dir is empty so it is a no-op, and
        # on any relaunch it skips files that already produced a report.
        argv.append("--resume")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.log, "a", encoding="utf-8")
        fh.write(f"\n=== attempt {self.attempts} at {time.strftime('%H:%M:%S')} ===\n")
        fh.flush()
        self.proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                     cwd=str(HERE))

    def stalled_for(self) -> float:
        """Seconds since this shard last wrote to its log.

        The failure this catches: the OpenAI client can end up holding an
        ESTABLISHED TCP connection that passes no data, without its timeout
        firing. Observed in the wild at 4h54m and 4h12m stalls, 0% CPU, three
        leaked sockets on one process. The child never exits, so poll() returns
        None forever and ordinary retry logic never engages. Killing it clears
        the wedge instantly -- progress resumed within two minutes.
        """
        try:
            return time.time() - self.log.stat().st_mtime
        except OSError:
            return 0.0

    def tail(self, n: int = 4000) -> str:
        try:
            with open(self.log, encoding="utf-8", errors="replace") as f:
                return f.read()[-n:]
        except OSError:
            return ""


def merge(shard_out_dirs: List[Path], final_out: Path, passthrough: List[str],
          mode: str, master_seed: Optional[int], model_name: str,
          dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    """Combine shard reports into one report identical in shape to a single run."""
    files_dir = final_out / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    seen = set()
    for d in shard_out_dirs:
        for f in sorted((d / "files").glob("*.json")) if (d / "files").is_dir() else []:
            if f.name in seen:
                continue
            seen.add(f.name)
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  WARN: unreadable {f}: {e}")
                continue
            # Point `file` back at the dataset, not the shard's symlink dir.
            # Downstream table building treats this as the row's origin path, so
            # leaving it pointing into a temporary shard dir would break it.
            if dataset_dir is not None and r.get("file"):
                r["file"] = str(Path(dataset_dir) / Path(r["file"]).name)
            results.append(r)
            (files_dir / f.name).write_text(json.dumps(r, indent=2), encoding="utf-8")
        for extra in ("workspaces",):
            src = d / extra
            if src.is_dir():
                shutil.copytree(src, final_out / extra, dirs_exist_ok=True)

    summary = _ets._build_summary(results, sorted(seen), mode,
                                  global_tn=False, master_seed=master_seed,
                                  model_name=model_name)
    summary["_sharded"] = {"shards": len(shard_out_dirs), "files_merged": len(seen)}
    (final_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-dir", default="../dataset/final")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="Real failures tolerated per shard before giving up")
    ap.add_argument("--max-rate-limit-waits", type=int, default=12,
                    help="Rate-limit backoffs per shard; these do not consume attempts")
    ap.add_argument("--backoff", type=int, default=60, help="Initial backoff seconds")
    ap.add_argument("--max-backoff", type=int, default=1800)
    ap.add_argument("--stagger", type=float, default=3.0,
                    help="Seconds between shard launches, to avoid a thundering herd")
    ap.add_argument("--stall-timeout", type=int, default=900,
                    help="Kill and relaunch a shard whose log has been silent this "
                         "long (seconds, default 15m; 0 disables). Catches wedged "
                         "API connections that never time out on their own")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip launching; just merge whatever the shard output "
                         "dirs already hold into the final report. For salvaging "
                         "a run that was cut short (e.g. a wedged file that the "
                         "stall watchdog can only relaunch into forever)")
    ap.add_argument("--min-shards", type=int, default=4,
                    help="Floor for adaptive concurrency. Each rate-limit event "
                         "lowers the launch ceiling by one for --rl-window "
                         "seconds; the ceiling never drops below this")
    ap.add_argument("--rl-window", type=float, default=600.0,
                    help="Rolling window (seconds) over which rate-limit events "
                         "count against the launch ceiling before decaying")
    ap.add_argument("--poll-interval", type=float, default=60.0,
                    help="Seconds between supervisor poll passes")
    ap.add_argument("--seed-from", default=None,
                    help="Reuse per-file reports from a previous partial run of this "
                         "same arm, so only missing files are redone")
    ap.add_argument("--max-seeds", type=int, default=10,
                    help="Mirrors --N; used only to weight the shard balance")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER,
                    help="After --, args forwarded to evaluate_test_split.py")
    args = ap.parse_args()

    passthrough = [a for a in args.passthrough if a != "--"]
    if not passthrough:
        raise SystemExit("nothing to pass through; put evaluate_test_split args after --")
    for banned in ("--dir", "--output-dir", "--resume"):
        if banned in passthrough:
            raise SystemExit(f"{banned} is managed by this script; remove it")

    dataset = Path(args.dataset_dir)
    files = sorted(dataset.glob("*.json"))
    if not files:
        raise SystemExit(f"no dataset files in {dataset}")

    final_out = Path(args.output_dir)
    work = final_out.parent / f"_{final_out.name}_shards"

    if args.merge_only:
        out_dirs = sorted(work.glob("out_*"))
        if not out_dirs:
            raise SystemExit(f"--merge-only: no out_* dirs under {work}")
        seed = None
        if "--seed" in passthrough:
            seed = int(passthrough[passthrough.index("--seed") + 1])
        model = "unknown"
        if "--openai-model" in passthrough:
            model = passthrough[passthrough.index("--openai-model") + 1]
        summary = merge(out_dirs, final_out, passthrough, "regex", seed, model,
                        dataset_dir=args.dataset_dir)
        print(f"merge-only -> {final_out}/summary.json "
              f"({summary['_sharded']['files_merged']}/{len(files)} files, "
              f"recall {summary.get('average_true_positive_rate')})")
        return 0

    buckets = balance(files, args.shards, args.max_seeds)
    in_dirs = make_shard_dirs(buckets, work)
    reused = seed_shard_outputs(buckets, work, args.seed_from)

    print(f"{len(files)} files -> {len(in_dirs)} shards")
    if reused:
        print(f"  reusing {reused} completed file report(s) from {args.seed_from}; "
              f"{len(files) - reused} left to run")
    for i, b in enumerate(buckets):
        w = sum(estimate_cost(f, args.max_seeds) for f in b)
        print(f"  shard {i:02d}: {len(b):2d} files, ~{w:.0f} API calls "
              f"(~{w * 31 / 3600:.1f}h at 31s/call)")

    shards = [Shard(i, d, work / f"out_{i:02d}", passthrough, args.python,
                    HERE / "evaluate_test_split.py")
              for i, d in enumerate(in_dirs)]

    stall_timeout = args.stall_timeout
    gov = ConcurrencyGovernor(len(shards), args.min_shards, args.rl_window)

    def running_now() -> int:
        return sum(1 for x in shards
                   if x.proc is not None and x.proc.poll() is None)

    t0 = time.time()
    for s in shards:
        # Full width on a cold start (no events yet); the check matters only if
        # a limit is tripped mid-stagger, in which case the rest stay pending
        # and the poll loop launches them as the ceiling recovers.
        if not gov.can_launch(running_now()):
            break
        s.launch()
        time.sleep(args.stagger)
    print(f"\nlaunched {running_now()}/{len(shards)} shards "
          f"(ceiling {gov.limit()}, floor {gov.floor}); polling\n")

    try:
        while True:
            alive = 0
            pending: List[Shard] = []
            for s in shards:
                if s.done or s.failed:
                    continue
                if s.proc is None:
                    pending.append(s)
                    alive += 1
                    continue
                rc = s.proc.poll()
                if rc is None:
                    stalled = s.stalled_for()
                    if stall_timeout and stalled > stall_timeout:
                        print(f"[shard {s.idx:02d}] STALLED {stalled/60:.0f}m with no log "
                              f"output; killing to clear a wedged connection")
                        s.stalls += 1
                        try:
                            s.proc.kill()
                            s.proc.wait(timeout=20)
                        except Exception:
                            pass
                        s.proc = None
                        s.wake_at = time.time() + 5
                        # A stall is an environment fault, not a bad shard, so it
                        # does not consume an attempt.
                        s.attempts = max(0, s.attempts - 1)
                    alive += 1
                    continue

                if rc == 0:
                    s.done = True
                    print(f"[shard {s.idx:02d}] done ({s.n_done()}/{s.n_files()} files)")
                    continue

                blob = s.tail()
                if _rq.is_rate_limited(blob, rc):
                    s.rl_waits += 1
                    # Counts against the fleet-wide launch ceiling, not just
                    # this shard's own backoff.
                    gov.record()
                    if s.rl_waits > args.max_rate_limit_waits:
                        s.failed = True
                        print(f"[shard {s.idx:02d}] FAILED: rate-limit waits exhausted")
                        continue
                    wait = min(args.max_backoff,
                               args.backoff * (2 ** (s.rl_waits - 1)))
                    wait = int(wait * (0.8 + 0.4 * random.random()))  # jitter
                    s.proc = None
                    s.wake_at = time.time() + wait
                    print(f"[shard {s.idx:02d}] rate limited (rc={rc}); "
                          f"backing off {wait}s (wait {s.rl_waits})")
                    alive += 1
                    continue

                if s.attempts >= args.max_attempts:
                    s.failed = True
                    print(f"[shard {s.idx:02d}] FAILED after {s.attempts} attempts "
                          f"(rc={rc}); see {s.log}")
                    continue
                wait = min(args.max_backoff, args.backoff * s.attempts)
                s.proc = None
                s.wake_at = time.time() + wait
                print(f"[shard {s.idx:02d}] exited {rc}; retrying in {wait}s "
                      f"(attempt {s.attempts}/{args.max_attempts})")
                alive += 1

            now = time.time()
            report = gov.poll_report(now)
            if report:
                print(report)
            running = running_now()
            ceiling = gov.limit(now)
            held = 0
            # Oldest-ready first, so a shard held back does not starve.
            for s in sorted(pending, key=lambda x: x.wake_at):
                if now < s.wake_at:
                    continue
                if running >= ceiling:
                    held += 1
                    continue
                s.launch()
                running += 1
            if held:
                print(f"[width] holding {held} ready shard(s): {running} running "
                      f"at ceiling {ceiling}/{gov.total}")

            if alive == 0:
                break
            done_files = sum(s.n_done() for s in shards)
            total_files = sum(s.n_files() for s in shards)
            el = time.time() - t0
            rate = done_files / el * 3600 if el > 0 else 0
            eta = (total_files - done_files) / rate if rate > 0 else float("inf")
            print(f"  [{time.strftime('%H:%M:%S')}] {done_files}/{total_files} files, "
                  f"{alive} shard(s) active, {rate:.1f} files/h, "
                  f"ETA {eta:.1f}h" if math.isfinite(eta) else
                  f"  {done_files}/{total_files} files, {alive} active")
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\ninterrupted; terminating shards (progress is kept, rerun to resume)")
        for s in shards:
            if s.proc and s.proc.poll() is None:
                s.proc.terminate()
        for s in shards:
            if s.proc:
                try:
                    s.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    s.proc.kill()
        return 130

    ok = sum(1 for s in shards if s.done)
    stalls = sum(s.stalls for s in shards)
    print(f"\n{ok}/{len(shards)} shards finished, "
          f"{time.time() - t0:.0f}s elapsed"
          + (f", {stalls} stall(s) cleared" if stalls else ""))

    mode = "regex"
    seed = None
    if "--seed" in passthrough:
        seed = int(passthrough[passthrough.index("--seed") + 1])
    model = "unknown"
    if "--openai-model" in passthrough:
        model = passthrough[passthrough.index("--openai-model") + 1]

    summary = merge([s.out_dir for s in shards], final_out, passthrough,
                    mode, seed, model, dataset_dir=args.dataset_dir)
    print(f"merged -> {final_out}/summary.json")
    print(f"  files            : {summary['_sharded']['files_merged']}")
    print(f"  recall           : {summary.get('average_true_positive_rate')}")
    print(f"  test recall      : {summary.get('average_test_true_positive_rate')}")
    print(f"  api calls        : {summary.get('total_api_calls')}")
    print(f"  tokens           : {summary.get('total_tokens')}")
    return 0 if all(s.done for s in shards) else 1


if __name__ == "__main__":
    sys.exit(main())
