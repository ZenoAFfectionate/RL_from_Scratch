import importlib
import json
import math
import os
import sys
from datetime import datetime

from transformers import TrainerCallback

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use importlib to resolve the *top-level* utils.rewards module.
# A plain ``from utils.rewards import ...`` would resolve "utils" to this
# file itself (trl_trainer/utils.py) because Python checks the current
# package first, causing a circular self-import.
_rewards = importlib.import_module("utils.rewards")
dsr1_reward_fn = _rewards.dsr1_reward_fn
gsmk_reward_fn = _rewards.gsmk_reward_fn
code_reward_fn = _rewards.code_reward_fn
mmlu_reward_fn = _rewards.mmlu_reward_fn

_vllm_helper = importlib.import_module("utils.vllm_helper")
init_vllm = _vllm_helper.init_vllm
evaluate_vllm = _vllm_helper.evaluate_vllm
load_safetensors_into_vllm = _vllm_helper.load_safetensors_into_vllm


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



def get_base_reward_fn(dataset_name):
    """Return the appropriate base reward function for the dataset."""
    return {
        "math": dsr1_reward_fn,
        "gsmk": gsmk_reward_fn,
        "code": code_reward_fn,
        "mmlu": mmlu_reward_fn,
    }[dataset_name]


def make_trl_reward_fn(base_reward_fn):
    """
    Wrap the project's reward function to match TRL's expected interface.

    TRL calls: reward_func(prompts=..., completions=..., completion_ids=..., **kwargs)
    and expects a list of floats.

    The project's reward functions: fn(responses, ground_truths) -> list[dict]
    where each dict has a "reward" key.
    """
    def trl_reward_fn(prompts, completions, solution=None, **kwargs):
        if solution is None:
            return [0.0] * len(completions)

        results = base_reward_fn(completions, solution)
        return [r["reward"] for r in results]

    trl_reward_fn.__name__ = base_reward_fn.__name__
    return trl_reward_fn


class FileLoggingCallback(TrainerCallback):
    """Write enriched training metrics to ``<output_dir>/record.txt``.

    Automatically computes derived metrics (PPL from loss, tokens/s, etc.)
    and merges them with the trainer's native logs (which already include
    entropy, mean_token_accuracy, grad_norm, num_tokens from TRL/HF Trainer).

    Each line is a JSON object for easy parsing.
    """

    def __init__(self, output_dir):
        self.log_path = os.path.join(output_dir, "record.txt")
        os.makedirs(output_dir, exist_ok=True)
        self._start_time = None
        self._last_log_time = None
        self._last_tokens = 0
        # Write header
        with open(self.log_path, "w") as f:
            f.write(f"# Training log — started {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = datetime.now()
        self._last_log_time = self._start_time
        self._last_tokens = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        now = datetime.now()
        entry = {
            "step": state.global_step,
            "timestamp": now.strftime("%H:%M:%S"),
        }

        # Copy all native metrics from the trainer
        # (loss, grad_norm, learning_rate, entropy, mean_token_accuracy,
        #  num_tokens, epoch, rewards/*, etc.)
        for k, v in logs.items():
            entry[k] = v

        # --- Derived: PPL from loss ---
        loss = logs.get("loss") or logs.get("eval_loss")
        if loss is not None:
            try:
                ppl = math.exp(loss)
            except OverflowError:
                ppl = float("inf")
            key = "ppl" if "loss" in logs else "eval_ppl"
            entry[key] = round(ppl, 4)

        # --- Derived: tokens per second ---
        num_tokens = logs.get("num_tokens")
        if num_tokens is not None and self._last_log_time is not None:
            elapsed = (now - self._last_log_time).total_seconds()
            delta_tokens = num_tokens - self._last_tokens
            if elapsed > 0 and delta_tokens > 0:
                entry["tokens_per_second"] = round(delta_tokens / elapsed, 1)
            self._last_tokens = num_tokens
        self._last_log_time = now

        # --- Derived: elapsed time since training started ---
        if self._start_time is not None:
            entry["elapsed_min"] = round((now - self._start_time).total_seconds() / 60, 2)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_post_training_eval(model_path, dataset_name, eval_data_path,
                           prompt_template, output_filepath, seed=42,
                           model_id=None):
    """Run generation-based evaluation on the validation set using vLLM.

    Loads the *original* model into vLLM (so all weights—including the
    visual encoder—are present), then swaps in the fine-tuned text weights
    from the saved checkpoint.  This avoids vLLM's strict weight-validation
    error that would occur if we loaded a text-only checkpoint directly as
    a VL model.

    Parameters
    ----------
    model_path : str
        Path to the saved HuggingFace-format model directory (must contain
        ``model.safetensors``).
    dataset_name : str
        One of "math", "code", "gsmk", "mmlu".
    eval_data_path : str
        Path to the validation JSONL file.
    prompt_template : str
        The prompt template string.
    output_filepath : str
        Path to write per-example JSONL results.
    seed : int
        Random seed for vLLM.
    model_id : str or None
        Original HuggingFace model ID (e.g. "Qwen/Qwen3.5-2B").  Used to
        initialise vLLM so that *all* weights (including vision encoder) are
        present.  Falls back to loading directly from *model_path* if None.

    Returns
    -------
    dict
        Keys: accuracy, format_accuracy, answer_accuracy.
    """
    from vllm import SamplingParams

    # Load validation data and format prompts
    with open(eval_data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    prompts = []
    answers = []

    int_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    for item in raw_data:
        if dataset_name == "mmlu":
            prompt = prompt_template.format(
                subject=item["subject"],
                question=item["question"],
                options=item["choices"],
            )
            prompts.append(prompt)
            answers.append(int_to_letter[item["answer"]])
        elif dataset_name == "gsmk":
            prompt = prompt_template.format(question=item["problem"])
            prompts.append(prompt)
            answers.append(item["solution"])
        else:
            prompt = prompt_template.format(problem=item["problem"])
            prompts.append(prompt)
            if dataset_name == "code":
                answers.append(item.get("test", item.get("solution", "")))
            else:
                answers.append(item["solution"])

    # Initialize vLLM from the original model so ALL weights (including
    # vision encoder) are present and pass vLLM's strict validation.
    # Then swap in the fine-tuned text weights from the checkpoint.
    safetensors_file = os.path.join(model_path, "model.safetensors")
    if model_id and os.path.isfile(safetensors_file):
        print(f"\n>>> Initialising vLLM from original model: {model_id}")
        vllm_model = init_vllm(model_id, "cuda:0", seed)
        print(f">>> Loading fine-tuned weights from: {model_path}")
        load_safetensors_into_vllm(vllm_model, safetensors_file)
    else:
        # Fallback: load directly (works when all expected weights exist)
        print(f"\n>>> Loading model into vLLM: {model_path}")
        vllm_model = init_vllm(model_path, "cuda:0", seed)

    # Greedy decoding params — shorter for MMLU (short answers)
    if dataset_name == "mmlu":
        eval_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
        )
    else:
        eval_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            stop=["</answer>", "</solution>", "```"],
            include_stop_str_in_output=True,
        )

    # Get the reward function for this dataset
    reward_fn = get_base_reward_fn(dataset_name)

    # Run evaluation (evaluate_vllm internally frees GPU memory)
    print(">>> Running post-training evaluation...")
    metrics = evaluate_vllm(
        vllm_model, reward_fn, prompts, answers,
        eval_sampling_params, output_filepath,
    )

    return metrics
