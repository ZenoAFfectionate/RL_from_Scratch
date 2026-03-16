import os
import sys
import random
import argparse

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from utils.vllm_helper import init_policy, init_vllm
from algorithms.ppo import PPO_Trainer
from trainer.utils import load_prompt_template, load_jsonl, get_reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO training.")

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "code"])


    # PPO hyperparameters:
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_iterations", type=int, default=200, help="Number of PPO loop.")
    parser.add_argument("--rollout_epochs", type=int, default=2, help="PPO epochs per rollout batch.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of prompts per rollout batch.")
    parser.add_argument("--micro_batch", type=int, default=2, help="Number of prompts per micro-batch.")

    parser.add_argument("--clip_eps", type=float, default=0.2, help="Policy ratio clipping range.")
    parser.add_argument("--clip_value_eps", type=float, default=0.2, help="Value function clipping range.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--lambd", type=float, default=0.95, help="GAE lambda for advantage estimation.")
    parser.add_argument("--valuefn_coeff", type=float, default=0.50, help="Value func loss coefficient.")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient (0 = disabled).")

    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target_kl", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--value_head_lr", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--val_interval", type=int, default=50)
    parser.add_argument("--train_device", type=str, default="cuda:1")
    parser.add_argument("--valid_device", type=str, default="cuda:2")
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--max_generation_length", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--output_dir", type=str, default="../checkpoints/PPO")

    parser.add_argument("--wandb_project", type=str, default="PPO-MATH")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # -----------------------------------
    #  Load prompt template and datasets
    # -----------------------------------
    print(f">>> Loading prompt template for {args.dataset}")
    args.prompt_template = load_prompt_template(args.dataset)

    print(">>> Loading training and validation data...")
    train_data = load_jsonl(f'../data/{args.dataset}/train.jsonl')
    valid_data = load_jsonl(f'../data/{args.dataset}/valid.jsonl')

    reward_fn = get_reward_fn(args.dataset)

    # --------------------------------------------------
    #  Initialize policy model, optimizer and scheduler
    # --------------------------------------------------
    print(f">>> Initializing Policy Model: {args.model_id}")
    policy, tokenizer = init_policy(args.model_id, args.train_device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    
    # Constant LR: RL algorithms use clipping + KL penalty to control
    # update magnitude, so cosine decay is unnecessary and can hurt
    # learning in later iterations when the data distribution shifts.
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    print(">>> Initializing validation model...")
    valid_model = init_vllm(args.model_id, args.valid_device, args.seed)

    # -------------------------------------
    #  Create Trainer and Run PPO Training
    # -------------------------------------
    trainer = PPO_Trainer(
        train_model=policy,
        valid_model=valid_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_data,
        valid_dataset=valid_data,
        args=args,
        reward_fun=reward_fn,
    )
    trainer.train()
