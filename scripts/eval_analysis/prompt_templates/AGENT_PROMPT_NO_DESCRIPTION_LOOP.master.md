<!-- MASTER TEMPLATE.
     Substitute:
       {{EVAL_ROOT}}   absolute path of the run directory the agent works in
       {{AGENT}}       the coding agent's name as addressed in the prompt, e.g. Claude or Codex
       {{RESULTS_DIR}} directory the agent writes batch results to, e.g.
                       claude_results_loop (Claude) or codex_results_loop (Codex)
     The Claude and Codex prompts used in the paper differ ONLY in these two
     substitutions; everything else is identical. The instantiated pairs are in
     this directory alongside this file. -->

# Agent prompts for batch regex-rule generation (NO description)

Variant of `AGENT_PROMPT.md` that uses inputs with the `description` field stripped out, so the agent must infer the malicious pattern from the true-positive log alone.

## Orchestrator prompt (paste into the {{AGENT}} session at `{{EVAL_ROOT}}`)

> Spawn 10 agents in parallel (one message with 10 Agent tool calls). Each agent gets the per-agent prompt below, with `{BATCH}` substituted by the three-digit batch number for that agent — `001` through `010`. Use `subagent_type: "general-purpose"`. Do not run them sequentially; they are independent.
>
> When all 10 return, confirm each wrote its output file under `{{EVAL_ROOT}}/{{RESULTS_DIR}}/batch_{BATCH}_result.json` and report any failures.

---

## Per-agent prompt (substitute `{BATCH}`)

You are processing batch `{BATCH}` in `{{EVAL_ROOT}}/`. Work there as your CWD.

### Inputs
- `rep_01_no_description/batch_{BATCH}/*.json` — files are named numerically (`01.json`, `02.json`, …, up to `66.json`). Each file has shape:
  ```json
  { "true_positive": "<one log line>" }
  ```
  The log line is your **only** signal about what the rule is trying to detect. There is no description, and the numeric filename carries no semantic hint either.
- `randomized/<same_filename>.json` — a JSON array of log strings mixing true positives and true negatives. Labels are NOT exposed; you don't know which is which. Filenames here use the same numeric scheme, so input `batch_{BATCH}/07.json` corresponds to `randomized/07.json`.

### Task
For each input JSON in `rep_01_no_description/batch_{BATCH}/`:

1. Read `true_positive` from the input JSON.
2. From the log alone, infer the likely malicious indicator(s) — suspicious binary names, command-line arguments, registry paths, network destinations, event IDs, etc.
3. Craft a Python regex that captures that indicator. Use the `true_positive` as a concrete example of what must match.
4. Sanity-check: `re.search(regex, true_positive)` must succeed. If it doesn't, revise the regex before proceeding.
5. Load the matching `randomized/<same_filename>.json` (it's a JSON list of log strings).
6. Run `re.search(regex, log)` against **every** log in that list. Collect the logs that actually matched.
7. **LLM-judged refinement loop.** For every log in the matched set, you ({{AGENT}}) read the log and decide whether it actually represents the malicious behavior the `true_positive` exemplifies — i.e., whether it is a true positive or a false positive that the regex incorrectly captured.
   - If **all** matches look malicious (consistent with the true-positive pattern), keep the regex.
   - If **any** match is benign / not the targeted malicious behavior, the regex is too permissive. Revise it so that:
     - it still matches the original `true_positive`, and
     - it no longer matches the benign log(s) you flagged.
   - Re-run step 6 with the new regex and re-judge. Repeat until either (a) every match is judged malicious, or (b) you have iterated 25 times — whichever comes first. Record the final regex and final matching set.
   - You do **not** need to spawn a separate agent for the judging; do the malicious/benign call yourself, in-context, log by log. Be conservative: if you cannot tell, treat it as a false positive and tighten.

**Do not hallucinate matches.** Execute the regex. If you list a log under `matching_logs`, it must be a literal hit from `re.search` *with the final regex*.

Recommended approach: write one Python script that walks the batch folder and, for each file, executes the regex against the pool. The judging step (step 7) is *not* something the script can do — you must read the matched logs yourself and reason about each one before deciding to revise the regex. A reasonable workflow: run the script, read its per-file match output, judge, hand-edit the regex for any file that needs it, re-run, repeat.

### Output

Write ONE consolidated file to:
```
{{EVAL_ROOT}}/{{RESULTS_DIR}}/batch_{BATCH}_result.json
```

Shape:
```json
{
  "batch": "batch_{BATCH}",
  "results": [
    {
      "name": "<source numeric filename without .json, e.g. \"07\">",
      "true_positive_example": "...",
      "regex": "...",
      "matching_logs": ["<log>", "<log>", "..."],
      "refinement_iterations": <int, number of regex revisions made during the LLM-judging loop, 0 if the first regex passed>,
      "rejected_false_positives": ["<benign log that an earlier regex matched and was tightened out>", "..."]
    }
  ]
}
```

### Constraints
- Use Python's `re` module. If logs are XML or multiline, use `re.DOTALL` as needed.
- Escape backslashes properly for JSON (`\\` in the file).
- Prefer regexes that are specific enough to avoid obvious false positives but permissive enough to generalize beyond the single TP example (e.g., wildcards for timestamps, GUIDs, host names).
- If a regex fails to match its own `true_positive`, that's a bug — fix it before writing output.
- Process every file in your batch. If any file fails, include it with `"regex": null` and `"matching_logs": []` plus an `"error": "<why>"` field rather than skipping.

- Write any helper scripts or intermediate files under `scratch/batch_{BATCH}/` (relative to the eval CWD). Do not use `/tmp/` or any path shared with other batches.

### Report back
Give a short summary: how many files processed, how many regexes matched ≥1 log, how many required refinement iterations (and how many hit the 25-iteration cap), any files where the regex failed to match its own TP, and the output path.
