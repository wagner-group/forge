# FoRGe: Feedback-Oriented Rule Generation for Security Log Detection

Research artifact for the ISC 2026 paper.

FoRGe is a tool-calling LLM agent that synthesises regular-expression detection
rules from a single example log line and an iterative feedback loop, evaluated
against an off-the-shelf coding agent as a baseline.

## Contents

- `splunk_rule_testing/` — the FoRGe harness, the baseline agent drivers, the
  detection engines, and the train/test splitter.
- `scripts/` — the scoring library, the verification scripts, and the prompt
  templates given to the agents.
- `dataset/final/` — the benchmark: 66 detection tasks, 19,428 log lines.
- `dataset/filename_mapping.md` — slot number to rule name, used by the Claude
  scorers. Derived from `dataset/final/`; `scripts/make_filename_mapping.py`
  regenerates and verifies it.
- `env.example` — the environment variables the agents read, grouped by
  which part of the code uses them.

## Experimental data

Every generated rule and every per-trial result behind the published tables is
attached to the latest release as four per-arm archives — about 195 MB
compressed, 1.6 GB and 20,059 files unpacked:

<https://github.com/wagner-group/forge/releases/latest>

Unpack them as `results/` at the root of this repository (`SHA256SUMS.txt` in the
release verifies the downloads). Scripts that need the data exit with a message
saying so when it is absent.

## Verifying the paper

```bash
pip install -r requirements.txt       # or: conda env create -f environment.yml
python3 scripts/smoke_test.py                       # sub-second integrity check
python3 scripts/verify_published_cells.py           # Tables 3 and 5
python3 scripts/verify_published_cells.py --dataset # Tables 1 and 6
python3 scripts/verify_table4.py                    # Table 4 + Section 5.1 deltas
python3 scripts/verify_appendix.py                  # Appendix A
```

All 57 published values across Tables 1, 3, 4, 5 and 6, the Section 5.1 deltas
and Appendix A reproduce from this artifact, matching the published values to one decimal
place or better.

TPR is averaged over 55 of the 66 tasks and FPR over 62. The excluded tasks
cannot be scored for that metric: 10 tasks have a single positive, which is used
as the seed and so leaves nothing to score; 3 tasks have no negatives in the set
being evaluated; and one task, `linux_auditd_disable_or_modify_system_firewall`,
had two labels corrected after some runs had already finished, so it is left out
of both metrics.

## Models

The paper reports FoRGe on two models: Gemini 3.1 Pro Preview and GPT-5.6 Luna.
Gemini gives the better results.

The baselines are Claude Code, running Claude Opus 4.7 at maximum reasoning
effort, and the OpenAI Codex CLI, running GPT-5.6 Luna at `effort=none`.

## Dataset provenance

The 66 tasks in `dataset/` are derived from
[Splunk Security Content](https://github.com/splunk/security_content) and
[Splunk Attack Data](https://github.com/splunk/attack_data), both distributed
under the Apache License 2.0, and are redistributed here. Tasks record its
upstream detection rule in `source.rule_source` when relevant.

The log lines are not byte-identical to upstream. Hostnames and MITRE technique
identifiers that were artifacts of the attack simulations were rewritten before
publication to prevent label leakage.

## License and citation

MIT — see [`LICENSE`](LICENSE), whose scope section covers the redistributed
third-party content. To cite, see [`CITATION.cff`](CITATION.cff).
