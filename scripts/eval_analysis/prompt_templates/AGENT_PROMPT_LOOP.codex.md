# Agent prompts for batch regex-rule generation (with description, with refinement loop)

Variant of the with-description prompt that adds an LLM-judged refinement loop on top of the regex matches, so the agent tightens regexes that capture benign logs.

## Orchestrator prompt (paste into the Codex session at `{{EVAL_ROOT}}`)

> Spawn 10 agents in parallel (one message with 10 Agent tool calls). Each agent gets the per-agent prompt below, with `{BATCH}` substituted by the three-digit batch number for that agent — `001` through `010`. Use `subagent_type: "general-purpose"`. Do not run them sequentially; they are independent.
>
> When all 10 return, confirm each wrote its output file under `{{EVAL_ROOT}}/codex_results_loop/batch_{BATCH}_result.json` and report any failures.

---

## Per-agent prompt (substitute `{BATCH}`)

You are processing batch `{BATCH}` in `{{EVAL_ROOT}}/`. Work there as your CWD.

### Inputs
- `single_positive_log_with_description/batch_{BATCH}/*.json` — files are named numerically (`01.json`, `02.json`, …, up to `66.json`). Each file has shape:
  ```json
  { "description": "...", "true_positive": "<one log line>" }
  ```
  The numeric filename carries no semantic hint about the rule — treat it purely as an identifier.
- `randomize_pre_filter/<same_filename>.json` — a JSON array of log strings mixing true positives and true negatives. Labels are NOT exposed; you don't know which is which. Filenames use the same numeric scheme, so input `batch_{BATCH}/07.json` corresponds to `randomize_pre_filter/07.json`.

### Task
For each input JSON in `single_positive_log_with_description/batch_{BATCH}/`:

1. Read `description` and `true_positive` from the input JSON.
2. Craft a Python regex that captures the malicious activity described, using the `true_positive` as a concrete example of what must match.
3. Sanity-check: `re.search(regex, true_positive)` must succeed. If it doesn't, revise the regex before proceeding.
4. Load the matching `randomize_pre_filter/<same_filename>.json` (it's a JSON list of log strings).
5. Run `re.search(regex, log)` against **every** log in that list. Collect the logs that actually matched.
6. **LLM-judged refinement loop.** For every log in the matched set, you (Codex) read the log and decide whether it actually represents the malicious behavior described — i.e., whether it is a true positive or a false positive that the regex incorrectly captured.
   - If **all** matches look malicious (consistent with the description and the true-positive pattern), keep the regex.
   - If **any** match is benign / not the targeted malicious behavior, the regex is too permissive. Revise it so that:
     - it still matches the original `true_positive`, and
     - it no longer matches the benign log(s) you flagged.
   - Re-run step 5 with the new regex and re-judge. Repeat until either (a) every match is judged malicious, or (b) you have iterated 25 times — whichever comes first. Record the final regex and final matching set.
   - You do **not** need to spawn a separate agent for the judging; do the malicious/benign call yourself, in-context, log by log. Be conservative: if you cannot tell, treat it as a false positive and tighten.

**Do not hallucinate matches.** Execute the regex. If you list a log under `matching_logs`, it must be a literal hit from `re.search` *with the final regex*.

Recommended approach: write one Python script that walks the batch folder and, for each file, executes the regex against the pool. The judging step (step 6) is *not* something the script can do — you must read the matched logs yourself and reason about each one before deciding to revise the regex. A reasonable workflow: run the script, read its per-file match output, judge, hand-edit the regex for any file that needs it, re-run, repeat.

### Output

Write ONE consolidated file to:
```
{{EVAL_ROOT}}/codex_results_loop/batch_{BATCH}_result.json
```

Shape:
```json
{
  "batch": "batch_{BATCH}",
  "results": [
    {
      "name": "<source numeric filename without .json, e.g. \"07\">",
      "description": "...",
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
- Write any helper scripts or intermediate files under `scratch/batch_{BATCH}/` (relative to the eval CWD). Do not use `/tmp/` or any path shared with other batches.
- Use Python's `re` module. If logs are XML or multiline, use `re.DOTALL` as needed.
- Escape backslashes properly for JSON (`\\` in the file).
- Prefer regexes that are specific enough to avoid obvious false positives but permissive enough to generalize beyond the single TP example (e.g., wildcards for timestamps, GUIDs, host names).
- If a regex fails to match its own `true_positive`, that's a bug — fix it before writing output.
- Process every file in your batch. If any file fails, include it with `"regex": null` and `"matching_logs": []` plus an `"error": "<why>"` field rather than skipping.

### Report back
Give a short summary: how many files processed, how many regexes matched ≥1 log, how many required refinement iterations (and how many hit the 25-iteration cap), any files where the regex failed to match its own TP, and the output path.
