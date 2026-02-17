import os
import sys
import random
import argparse

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from data.lm_dataset import RLHFDataset
from algorithms.dpo import DPO_Trainer
from utils.vllm_helper import init_policy
from trainer.utils import load_prompt_template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO training on RLHF preference data.")
    # model and dataset:
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="rlhf", choices=["rlhf"])
    # DPO hyperparameters:
    parser.add_argument("--beta", type=float, default=0.2, help="DPO temperature parameter.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Total batch size.")
    parser.add_argument("--micro_batch", type=int, default=4, help="Micro batch size.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length.")
    # logging and saving:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20,  help="Log interval.")
    parser.add_argument("--val_interval", type=int, default=200, help="Val interval.")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval.")
    parser.add_argument("--policy_device", type=str, default="cuda:1")
    parser.add_argument("--ref_device", type=str, default="cuda:2")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/DPO")
    parser.add_argument("--wandb_project", type=str, default="DPO-RLHF")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ----------------------------------
    #  Load prompt template and dataset
    # ----------------------------------
    print(f">>> Loading prompt template for {args.dataset}")
    template = load_prompt_template(args.dataset)

    print(f"\n>>> Loading training data...")
    train_dataset = RLHFDataset(f'../data/{args.dataset}/train.jsonl')
    print(f"    Loaded {len(train_dataset)} training examples")

    print(f">>> Loading validation data...")
    valid_dataset = RLHFDataset(f'../data/{args.dataset}/valid.jsonl')
    print(f"    Loaded {len(valid_dataset)} validation examples")

    # ----------------------------------
    #  Initialize models and tokenizer
    # ----------------------------------
    print(f">>> Initializing Policy Model: {args.model_id}")
    policy_model, tokenizer = init_policy(args.model_id, args.policy_device)
    policy_model.gradient_checkpointing_enable()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f">>> Initializing Reference Model: {args.model_id}")
    reference_model, _ = init_policy(args.model_id, args.ref_device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95), fused=True)

    gradient_accumulation_steps = args.batch_size // args.micro_batch
    total_micro_batches = len(train_dataset) // args.micro_batch
    total_steps = total_micro_batches * args.epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=int(total_steps * 0.05),
    )

    # ---------------------------------------
    #  Create Trainer and RLHF Training Loop
    # ---------------------------------------
    trainer = DPO_Trainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        template=template,
        args=args,
    )
    trainer.train()
    