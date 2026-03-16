"""
TRL-based GRPO Trainer for math/code reasoning tasks.

Functionally equivalent to trainer/grpo_trainer.py.

Usage examples:
  # GRPO on math dataset
  python grpo_trainer.py --dataset math --model_id Qwen/Qwen3.5-2B

  # GRPO on code dataset
  python grpo_trainer.py --dataset code --model_id Qwen/Qwen3.5-2B

  # With SFT warm-start
  python grpo_trainer.py --dataset math --init_checkpoint ../checkpoints/SFT4math/sft_final.pt
"""

import os
import sys
import json
import random
import argparse
from typing import List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import GRPOTrainer, GRPOConfig

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from trl_trainer.utils import load_prompt_template, get_base_reward_fn, make_trl_reward_fn as make_reward_fn, FileLoggingCallback, run_post_training_eval


def build_grpo_dataset(data_path, prompt_template, dataset_name):
    """
    Build a HuggingFace Dataset for GRPO training.

    TRL's GRPOTrainer expects a "prompt" column. We also include the ground
    truth solution/test for the reward function to use.

    Data format: {"problem": ..., "solution": ..., "test": ...}
    """
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    prompts = []
    solutions = []

    for item in raw_data:
        if dataset_name == "gsmk":
            prompt = prompt_template.format(question=item["problem"])
        else:
            prompt = prompt_template.format(problem=item["problem"])
        prompts.append(prompt)

        if dataset_name == "code":
            solutions.append(item.get("test", item.get("solution", "")))
        else:
            solutions.append(item["solution"])

    return Dataset.from_dict({
        "prompt": prompts,
        "solution": solutions,
    })


def get_reward_fn(dataset_name):
    """Return the appropriate TRL-compatible reward function for the dataset."""
    base_fn = get_base_reward_fn(dataset_name)
    return make_reward_fn(base_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRL-based GRPO Trainer")

    # --- Model ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "code", "gsmk"])
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Path to a .pt state_dict to warm-start the policy (e.g., SFT checkpoint).")
    parser.add_argument("--prompt_template_path", type=str, default=None)

    # --- GRPO Hyperparameters ---
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_iterations", type=int, default=2,
                        help="Number of GRPO update iterations per batch (mu in the paper).")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of completions per prompt (group size G).")
    parser.add_argument("--max_steps", type=int, default=250,
                        help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-device training batch size (number of prompts per step).")
    parser.add_argument("--micro_batch", type=int, default=2,
                        help="Per-device batch size for gradient accumulation.")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient.")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO-style clipping range.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--sampling_temperature", type=float, default=0.8)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=1024)

    # --- Logging & Saving ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    # ─── Set defaults ───────────────────────────────────────────────
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, f"checkpoints/[TRL]GRPO4{args.dataset}")
    if args.wandb_project is None:
        args.wandb_project = f"[TRL]GRPO-{args.dataset.upper()}"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ─── Load Model and Tokenizer ───────────────────────────────────
    print(f">>> Loading model and tokenizer: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT checkpoint if provided (warm-start)
    if args.init_checkpoint is not None:
        print(f">>> Loading initial checkpoint: {args.init_checkpoint}")
        state_dict = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        del state_dict
        print("    Checkpoint loaded successfully.")

    # ─── Build Dataset ──────────────────────────────────────────────
    prompt_template = load_prompt_template(args.dataset, args.prompt_template_path)

    train_path = os.path.join(project_root, f"data/{args.dataset}/train.jsonl")
    eval_path = os.path.join(project_root, f"data/{args.dataset}/valid.jsonl")
    print(f">>> Building GRPO datasets from {train_path}")

    train_dataset = build_grpo_dataset(train_path, prompt_template, args.dataset)
    eval_dataset = build_grpo_dataset(eval_path, prompt_template, args.dataset)
    print(f"    Train: {len(train_dataset)} prompts, Eval: {len(eval_dataset)} prompts")

    # ─── Reward Function ────────────────────────────────────────────
    reward_fn = get_reward_fn(args.dataset)

    # ─── Configure TRL GRPOTrainer ──────────────────────────────────
    gradient_accumulation_steps = args.batch_size // args.micro_batch
    run_name = f"trl_grpo_{args.dataset}_lr{args.lr}_beta{args.beta}"

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=args.clip_grad_norm,
        # GRPO-specific:
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.sampling_temperature,
        scale_rewards=True,
        # Logging:
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
        log_completions=True,
        num_completions_to_print=2,
        remove_unused_columns=False,
    )

    # ─── Initialize W&B ─────────────────────────────────────────────
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # ─── Create Trainer and Train ───────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[FileLoggingCallback(args.output_dir)],
    )

    print(">>> Starting TRL GRPO training...")
    trainer.train()

    # ─── Save Final Model ───────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f">>> Final model saved to {final_dir}")

    # Save .pt state_dict for compatibility with existing evaluation code
    pt_path = os.path.join(args.output_dir, f"trl_grpo_final.pt")
    torch.save(model.state_dict(), pt_path)
    print(f">>> State dict saved to {pt_path}")

    # ─── Post-Training Evaluation (generation-based) ───────────────
    import gc

    print("\n>>> Freeing training objects for post-training evaluation...")
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    eval_output = os.path.join(args.output_dir, "trl_grpo_eval_results.jsonl")
    metrics = run_post_training_eval(
        model_path=final_dir,
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
