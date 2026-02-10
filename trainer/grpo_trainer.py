import os
import sys
import json
import random
import argparse

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from utils.rewards import *
from utils.vllm_helper import init_policy, init_vllm
from algorithms.grpo import GRPO_Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO on Qwen-Math model.")

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "code"])
    parser.add_argument("--problem_key",  type=str, default="problem")
    parser.add_argument("--solution_key", type=str, default="solution")

    # GRPO hyperparameters:
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_iterations", type=int, default=250, help="Number of GRPO steps.")
    parser.add_argument("--rollout_epochs", type=int, default=2, help="Epochs per rollout batch.")
    parser.add_argument("--batch_size",  type=int, default=32, help="Total rollout batch size.")
    parser.add_argument("--micro_batch", type=int, default=4, help="Micro batch size.")
    parser.add_argument("--num_generations", type=int, default=8, help="Responses per prompt.")
    parser.add_argument("--beta", type=float, default=0.05, help="KL penalty coefficient.")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO-style clipping range.")
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--max_generation_length", type=int, default=2048)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=50)
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--valid_device", type=str, default="cuda:1")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/GRPO")

    parser.add_argument("--wandb_project", type=str, default="GRPO-MATH")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f">>> Loading prompt template...")
    with open(f'./prompts/{args.dataset}.prompt', "r") as f:
        args.prompt_template = f.read()

    print(">>> Loading training and validation data...")
    with open(f'../data/{args.dataset}/train.jsonl', "r") as f:
        train_data = [json.loads(line) for line in f]
    with open(f'../data/{args.dataset}/valid.jsonl', "r") as f:
        valid_data = [json.loads(line) for line in f]

    reward_fun = {
        "math": dsr1_reward_fn,
        "gsmk": gsmk_reward_fn,
        "code": code_reward_fn,
    }[args.dataset]


    # --------------------------------------------------
    #  Initialize policy model, optimizer and scheduler
    # --------------------------------------------------
    print(f">>> Initializing Policy Model: {args.model_id}")
    policy, tokenizer = init_policy(args.model_id, args.train_device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)

    training_steps = args.num_iterations * args.rollout_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05*training_steps),
        num_training_steps=training_steps,
    )

    print("Initializing validation model...")
    valid_model = init_vllm(args.model_id, args.valid_device, args.seed)

    # --------------------------------------
    #  Create Trainer and Run GRPO Training
    # --------------------------------------
    trainer = GRPO_Trainer(
        train_model=policy,
        valid_model=valid_model,
        tokenizer=tokenizer,
        reward_fun=reward_fun,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_data,
        valid_dataset=valid_data,
        args=args,
    )
    trainer.train()
