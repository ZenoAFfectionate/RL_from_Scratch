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
from algorithms.grpo import GRPO_Trainer
from trainer.utils import load_prompt_template, load_jsonl, get_reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO on Qwen-Math model.")

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "code"])


    # GRPO hyperparameters:
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_iterations", type=int, default=250, help="Number of GRPO steps.")
    parser.add_argument("--rollout_epochs", type=int, default=2, help="Epochs per rollout batch.")
    parser.add_argument("--batch_size",  type=int, default=32, help="Total rollout batch size.")
    parser.add_argument("--micro_batch", type=int, default=2, help="Micro batch size.")
    parser.add_argument("--num_generations", type=int, default=8, help="Responses per prompt.")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient.")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO-style clipping range.")
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--sampling_temperature", type=float, default=0.8)
    parser.add_argument("--max_generation_length", type=int, default=4096)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=50)
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--valid_device", type=str, default="cuda:1")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/GRPO")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Path to a .pt state_dict to warm-start the policy (e.g., SFT checkpoint).")

    parser.add_argument("--wandb_project", type=str, default="GRPO-MATH")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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

    # Load SFT checkpoint if provided (warm-start)
    if args.init_checkpoint is not None:
        print(f">>> Loading initial checkpoint: {args.init_checkpoint}")
        state_dict = torch.load(args.init_checkpoint, map_location=args.train_device)
        policy.load_state_dict(state_dict)
        del state_dict
        print("    Checkpoint loaded successfully.")
    
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)

    # Constant LR: RL algorithms use clipping + KL penalty to control
    # update magnitude, so cosine decay is unnecessary and can hurt
    # learning in later iterations when the data distribution shifts.
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    print("Initializing validation model...")
    valid_model = init_vllm(args.model_id, args.valid_device, args.seed)

    # --------------------------------------
    #  Create Trainer and Run GRPO Training
    # --------------------------------------
    trainer = GRPO_Trainer(
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
