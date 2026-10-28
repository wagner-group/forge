# FoRGe: Feedback-Oriented Rule Generation for Security Log Detection

Research artifact for the ISC 2026 paper.

FoRGe is a tool-calling LLM agent that synthesises regular-expression detection
rules from a single example log line and an iterative feedback loop, evaluated
against an off-the-shelf coding agent as a baseline.

## Contents

- Dataset / benchmark
- License
- Citation metadata

## What will be added

- Code
- Experimental data

## Dataset provenance

The 66 tasks in `dataset/` are derived from
[Splunk Security Content](https://github.com/splunk/security_content) and
[Splunk Attack Data](https://github.com/splunk/attack_data), both distributed
under the Apache License 2.0, and are redistributed here. Each task records its
upstream detection rule in `source.rule_source`.

The log lines are not byte-identical to upstream. Hostnames and MITRE technique
identifiers that were artifacts of the attack simulations were rewritten before
publication to prevent label leakage.
