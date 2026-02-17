"""
Shared utilities for custom trainers.

Contains prompt template loading, data loading, and reward function helpers
used across sft4chat_trainer, sft4reas_trainer, grpo_trainer, dpo_trainer, and ppo_trainer.
"""

import os
import sys
import json

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from utils.rewards import dsr1_reward_fn, gsmk_reward_fn, code_reward_fn

PROMPTS_DIR = os.path.join(project_root, "prompts")


def load_prompt_template(dataset_name, prompt_template_path=None):
    """
    Load a prompt template by dataset name from prompts/<dataset_name>.prompt.

    Args:
        dataset_name: Name of the dataset (e.g., "math", "code", "gsmk", "rlhf").
        prompt_template_path: Optional explicit path to a template file.

    Returns:
        The prompt template string.
    """
    if prompt_template_path and os.path.exists(prompt_template_path):
        with open(prompt_template_path, "r") as f:
            return f.read()

    prompt_path = os.path.join(PROMPTS_DIR, f"{dataset_name}.prompt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            return f.read()

    raise FileNotFoundError(
        f"Prompt template '{dataset_name}' not found at {prompt_path}."
    )


def load_jsonl(path):
    """Load a JSONL file and return a list of dicts."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def get_reward_fn(dataset_name):
    """Return the appropriate reward function for the dataset."""
    return {
        "math": dsr1_reward_fn,
        "gsmk": gsmk_reward_fn,
        "code": code_reward_fn,
    }[dataset_name]
