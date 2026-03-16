"""
TRL-based PPO Trainer for math/code reasoning tasks.

Functionally equivalent to trainer/ppo_trainer.py.

TRL's PPOTrainer expects an nn.Module reward model (AutoModelForSequenceClassification
style) rather than a callable reward function. This module provides a
`VerifiableRewardModel` wrapper that:
  - Accepts token IDs through the standard TRL `get_reward` interface
  - Internally decodes them and evaluates with the project's verifiable reward
    functions (dsr1_reward_fn, code_reward_fn, etc.)
  - Returns scalar rewards as if they were model logits

Usage examples:
  # PPO on math dataset
  python ppo_trainer.py --dataset math --model_id Qwen/Qwen3.5-2B

  # PPO on code dataset
  python ppo_trainer.py --dataset code --model_id Qwen/Qwen3.5-2B

  # With SFT warm-start
  python ppo_trainer.py --dataset math --init_checkpoint ../checkpoints/SFT4math/sft_final.pt
"""

import os
import sys
import json
import random
import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from trl_trainer.utils import load_prompt_template, get_base_reward_fn, FileLoggingCallback, run_post_training_eval


# ──────────────────────────────────────────────────────────────────────
#  Verifiable Reward Model wrapper for TRL PPOTrainer
# ──────────────────────────────────────────────────────────────────────

class VerifiableRewardModel(nn.Module):
    """
    A wrapper that makes verifiable reward functions compatible with
    TRL's PPOTrainer `get_reward()` interface.

    TRL's `get_reward()` calls:
        lm_backbone = getattr(model, model.base_model_prefix)
        output = lm_backbone(input_ids=..., attention_mask=..., ...)
        reward_logits = model.score(output.hidden_states[-1])

    This wrapper:
      1. Stores ground truth solutions keyed by prompt text
      2. When `score()` is called, decodes the input tokens, splits into
         prompt/response, looks up the ground truth, and evaluates
      3. Returns rewards as logits matching the expected shape
    """

    base_model_prefix = "backbone"

    def __init__(self, tokenizer, reward_fn, ground_truths: dict):
        """
        Args:
            tokenizer: The tokenizer used for decoding token IDs.
            reward_fn: The verifiable reward function (e.g., dsr1_reward_fn).
            ground_truths: Dict mapping prompt text -> ground truth solution.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ground_truths = ground_truths
        self._cached_input_ids = None

        # Dummy backbone that stores hidden_states (just passes through)
        self.backbone = DummyBackbone()
        # Dummy score layer (required by get_reward interface)
        self._score = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self._score.weight)

        # Config-like object for compatibility
        self.config = type("Config", (), {"pad_token_id": tokenizer.pad_token_id})()

    def score(self, hidden_states):
        """
        Called by TRL's get_reward() with hidden_states from the backbone.
        We ignore hidden_states and instead use the cached input_ids.
        """
        if self._cached_input_ids is None:
            # Fallback: return zeros
            B = hidden_states.shape[0]
            return torch.zeros(B, hidden_states.shape[1], 1, device=hidden_states.device)

        input_ids = self._cached_input_ids
        B, T = input_ids.shape
        device = input_ids.device

        # Decode all sequences
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        rewards = []
        for text in texts:
            # Try to find matching ground truth
            best_gt = None
            for prompt_key, gt in self.ground_truths.items():
                if prompt_key in text:
                    best_gt = gt
                    # Extract response (everything after the prompt)
                    response = text[text.index(prompt_key) + len(prompt_key):]
                    break

            if best_gt is not None:
                result = self.reward_fn([response], [best_gt])
                rewards.append(result[0]["reward"])
            else:
                rewards.append(0.0)

        # Return rewards as (B, T, 1) logits - reward placed at every position
        # TRL's get_reward extracts the value at the last non-pad position
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        reward_logits = reward_tensor.unsqueeze(1).unsqueeze(2).expand(B, T, 1)
        return reward_logits

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Direct forward pass (not typically used by get_reward)."""
        self._cached_input_ids = input_ids
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class DummyBackbone(nn.Module):
    """Minimal backbone that returns a dummy output compatible with get_reward."""

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        device = input_ids.device
        hidden = torch.zeros(B, T, 1, device=device, dtype=torch.float32)
        return type("Output", (), {
            "hidden_states": (hidden,),
            "last_hidden_state": hidden,
        })()


# ──────────────────────────────────────────────────────────────────────
#  Dataset and utility functions
# ──────────────────────────────────────────────────────────────────────

def build_ppo_dataset(data_path, prompt_template, dataset_name):
    """
    Build a HuggingFace Dataset for PPO training.

    TRL's PPOTrainer processes "input_ids" from the dataset. We provide
    the formatted prompt as "query" text that gets tokenized.

    Returns:
        dataset: HuggingFace Dataset with "query" column
        ground_truths: dict mapping formatted prompt -> ground truth
    """
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    queries = []
    ground_truths = {}

    for item in raw_data:
        if dataset_name == "gsmk":
            prompt = prompt_template.format(question=item["problem"])
        else:
            prompt = prompt_template.format(problem=item["problem"])
        queries.append(prompt)

        if dataset_name == "code":
            gt = item.get("test", item.get("solution", ""))
        else:
            gt = item["solution"]
        ground_truths[prompt] = gt

    return Dataset.from_dict({"query": queries}), ground_truths


def get_reward_fn(dataset_name):
    """Return the appropriate reward function for the dataset."""
    return get_base_reward_fn(dataset_name)


def tokenize_fn(examples, tokenizer, max_length):
    """Tokenize queries for PPO dataset."""
    return tokenizer(
        examples["query"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRL-based PPO Trainer")

    # --- Model ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "code", "gsmk"])
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Path to a .pt state_dict to warm-start the policy.")
    parser.add_argument("--prompt_template_path", type=str, default=None)

    # --- PPO Hyperparameters ---
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--total_episodes", type=int, default=6400,
                        help="Total number of episodes (prompts) to train on.")
    parser.add_argument("--num_ppo_epochs", type=int, default=2,
                        help="Number of PPO epochs per batch.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL penalty coefficient.")
    parser.add_argument("--cliprange", type=float, default=0.2,
                        help="Policy ratio clipping range.")
    parser.add_argument("--cliprange_value", type=float, default=0.2,
                        help="Value function clipping range.")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="Value function loss coefficient.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--response_length", type=int, default=4096,
                        help="Maximum response generation length.")
    parser.add_argument("--temperature", type=float, default=0.8)

    # --- Logging & Saving ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--max_prompt_length", type=int, default=1024)

    args = parser.parse_args()

    # ─── Set defaults ───────────────────────────────────────────────
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, f"checkpoints/[TRL]PPO4{args.dataset}")
    if args.wandb_project is None:
        args.wandb_project = f"[TRL]PPO-{args.dataset.upper()}"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ─── Load Model and Tokenizer ───────────────────────────────────
    print(f">>> Loading model and tokenizer: {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for PPO generation

    # Policy model with value head
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Load SFT checkpoint if provided
    if args.init_checkpoint is not None:
        print(f">>> Loading initial checkpoint: {args.init_checkpoint}")
        state_dict = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        policy_model.pretrained_model.load_state_dict(state_dict)
        del state_dict
        print("    Checkpoint loaded successfully.")

    # Reference model (frozen copy of policy)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if args.init_checkpoint is not None:
        state_dict = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        ref_model.pretrained_model.load_state_dict(state_dict)
        del state_dict

    # Value model (separate, shares architecture with policy)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # ─── Build Dataset ──────────────────────────────────────────────
    prompt_template = load_prompt_template(args.dataset, args.prompt_template_path)

    train_path = os.path.join(project_root, f"data/{args.dataset}/train.jsonl")
    eval_path = os.path.join(project_root, f"data/{args.dataset}/valid.jsonl")
    print(f">>> Building PPO datasets from {train_path}")

    train_dataset, train_gt = build_ppo_dataset(train_path, prompt_template, args.dataset)
    eval_dataset, eval_gt = build_ppo_dataset(eval_path, prompt_template, args.dataset)

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_prompt_length),
        batched=True,
        remove_columns=["query"],
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_prompt_length),
        batched=True,
        remove_columns=["query"],
    )
    print(f"    Train: {len(train_dataset)} prompts, Eval: {len(eval_dataset)} prompts")

    # ─── Reward Model ───────────────────────────────────────────────
    reward_fn = get_reward_fn(args.dataset)
    all_gt = {**train_gt, **eval_gt}
    reward_model = VerifiableRewardModel(tokenizer, reward_fn, all_gt)

    # ─── Configure TRL PPOTrainer ───────────────────────────────────
    gradient_accumulation_steps = args.batch_size // args.micro_batch
    run_name = f"trl_ppo_{args.dataset}_lr{args.lr}_kl{args.kl_coef}"

    ppo_config = PPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        total_episodes=args.total_episodes,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.lr,
        num_ppo_epochs=args.num_ppo_epochs,
        kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        vf_coef=args.vf_coef,
        gamma=args.gamma,
        lam=args.lam,
        max_grad_norm=args.clip_grad_norm,
        response_length=args.response_length,
        temperature=args.temperature,
        bf16=args.bf16,
        disable_tqdm=True,
        log_level="info",
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        report_to="wandb",
        gradient_checkpointing=True,
        stop_token="eos",
    )

    # ─── Initialize W&B ─────────────────────────────────────────────
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # ─── Create Trainer and Train ───────────────────────────────────
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[FileLoggingCallback(args.output_dir)],
    )

    print(">>> Starting TRL PPO training...")
    trainer.train()

    # ─── Save Final Model ───────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f">>> Final model saved to {final_dir}")

    # Save .pt state_dict for compatibility with existing evaluation code
    pt_path = os.path.join(args.output_dir, f"trl_ppo_final.pt")
    if hasattr(policy_model, "pretrained_model"):
        torch.save(policy_model.pretrained_model.state_dict(), pt_path)
    else:
        torch.save(policy_model.state_dict(), pt_path)
    print(f">>> State dict saved to {pt_path}")

    # ─── Post-Training Evaluation (generation-based) ───────────────
    import gc

    # Save the base model (without value head) for vLLM compatibility
    eval_model_dir = os.path.join(args.output_dir, "eval_model")
    print(f">>> Saving base model (without value head) for evaluation: {eval_model_dir}")
    if hasattr(policy_model, "pretrained_model"):
        policy_model.pretrained_model.save_pretrained(eval_model_dir)
    else:
        policy_model.save_pretrained(eval_model_dir)
    tokenizer.save_pretrained(eval_model_dir)

    print("\n>>> Freeing training objects for post-training evaluation...")
    del trainer
    del policy_model
    del ref_model
    del value_model
    del reward_model
    gc.collect()
    torch.cuda.empty_cache()

    eval_output = os.path.join(args.output_dir, "trl_ppo_eval_results.jsonl")
    metrics = run_post_training_eval(
        model_path=eval_model_dir,
        dataset_name=args.dataset,
        eval_data_path=eval_path,
        prompt_template=prompt_template,
        output_filepath=eval_output,
        seed=args.seed,
        model_id=args.model_id,
    )
    print(f"\n>>> Post-training evaluation results:")
    print(f"    Format Accuracy:  {metrics['format_accuracy']:.2f}%")
    print(f"    Answer Accuracy:  {metrics['answer_accuracy']:.2f}%")
    print(f"    Overall Accuracy: {metrics['accuracy']:.2f}%")
