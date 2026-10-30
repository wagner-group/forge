"""Materialize per-trial Claude Code workspaces with train-only data."""
from __future__ import annotations

import json
import os
import shutil
from typing import List


def prepare_trial_workspace(
    workspace_root: str,
    batch_id: str,
    numeric_id: str,
    seed_tp: str,
    train_pool: List[str],
) -> str:
    """Create minimal workspace for one Claude Code trial.

    Layout:
      workspace_root/single_positive_log_no_description/batch_{batch_id}/{NN}.json
      workspace_root/randomized/{NN}.json
      workspace_root/claude_results_no_description/
    """
    if os.path.exists(workspace_root):
        shutil.rmtree(workspace_root)
    os.makedirs(workspace_root, exist_ok=True)

    seed_dir = os.path.join(
        workspace_root,
        "single_positive_log_no_description",
        f"batch_{batch_id}",
    )
    os.makedirs(seed_dir, exist_ok=True)
    seed_file = os.path.join(seed_dir, f"{numeric_id}.json")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump({"true_positive": seed_tp}, f, indent=2)

    rand_dir = os.path.join(workspace_root, "randomized")
    os.makedirs(rand_dir, exist_ok=True)
    rand_file = os.path.join(rand_dir, f"{numeric_id}.json")
    with open(rand_file, "w", encoding="utf-8") as f:
        json.dump(train_pool, f, indent=2)

    os.makedirs(
        os.path.join(workspace_root, "claude_results_no_description"),
        exist_ok=True,
    )
    return workspace_root


def batch_result_path(workspace_root: str, batch_id: str) -> str:
    return os.path.join(
        workspace_root,
        "claude_results_no_description",
        f"batch_{batch_id}_result.json",
    )
