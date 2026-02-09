import os
import sys
import math
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
from utils.vllm_helper import *
from algorithms.sft4reas import SFT4ReasTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT on Qwen-Math model.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "gsmk", "code"])


    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling probability.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of tokens.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--valid_device", type=str, default="cuda:1")
    parser.add_argument("--wandb_project", type=str, default="SFT-Reason")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=50)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------
    #  Load Training and Validate Data
    # ---------------------------------
    print(f">>> Loading prompt template...")
    with open(f"./prompts/{args.dataset}.prompt", "r") as f:
        args.prompt_template = f.read()

    print(">>> Loading training and validation data...")
    train_path = f"../data/{args.dataset}/train.jsonl"
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]
    valid_path = f"../data/{args.dataset}/valid.jsonl"
    with open(valid_path, 'r') as f:
        valid_data = [json.loads(line) for line in f]

    reward_fun = {
        "math": dsr1_reward_fn,
        "gsmk": gsmk_reward_fn,
        "mmlu": mmlu_reward_fn,
        "code": code_reward_fn,
    }[args.dataset]

    # --------------------------------
    #  Initialize tokenizer and model
    # --------------------------------
    print(f"Loading model and tokenizer: {args.model_id}")
    model, tokenizer = init_policy(args.model_id, args.train_device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    grad_accum_steps = args.batch_size // args.micro_batch
    micro_batch_per_epoch = math.ceil(len(train_data) / args.micro_batch)
    training_steps = args.epochs * (micro_batch_per_epoch // grad_accum_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05*training_steps),
        num_training_steps=training_steps,
    )

    print("Initializing validation model...")
    valid_model = init_vllm(args.model_id, args.valid_device, args.seed)
    valid_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # ============================== #
    #  Create Trainer and Fine-Tune  #
    # ============================== #
    run_name = f"sft_{args.dataset}_lr{args.lr}"
    trainer = SFT4ReasTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        valid_data=valid_data,
        reward_fun=reward_fun,
        optimizer=optimizer,
        scheduler=scheduler,
        valid_model=valid_model,
        valid_sampling_params=valid_sampling_params,
        args=args,
        output_dir=f"../checkpoints/SFT4{args.dataset}",
        wandb_project=args.wandb_project,
        run_name=run_name,
    )
    trainer.train()
