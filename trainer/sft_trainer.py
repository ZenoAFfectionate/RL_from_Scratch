"""
Unified SFT Trainer for both Chat (ultrachat) and Reasoning (math/code/gsmk) datasets.

Merges the functionality of:
  - trainer/sft4chat_trainer.py  (--mode chat)
  - trainer/sft4reas_trainer.py  (--mode reasoning)

Usage examples:
  # Chat SFT on ultrachat dataset
  python sft_trainer.py --mode chat --model_id Qwen/Qwen3.5-2B

  # Reasoning SFT on math dataset
  python sft_trainer.py --mode reasoning --dataset math --model_id Qwen/Qwen3.5-2B

  # Reasoning SFT on code dataset
  python sft_trainer.py --mode reasoning --dataset code --model_id Qwen/Qwen3.5-2B
"""

import os
import sys
import math
import random
import argparse

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_schedule_with_warmup

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from utils.vllm_helper import init_policy, init_vllm, SamplingParams
from algorithms.sft4chat import SFT4ChatTrainer
from algorithms.sft4reas import SFT4ReasTrainer
from data.lm_dataset import InstructionTuningDataset
from trainer.utils import load_prompt_template, load_jsonl, get_reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified SFT Trainer (Chat + Reasoning)")

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--mode", type=str, default="chat",
                        choices=["chat", "reasoning"],
                        help="Training mode: 'chat' for ultrachat, 'reasoning' for math/code/gsmk.")
    parser.add_argument("--dataset", type=str, default="math",
                        choices=["math", "gsmk", "code"],
                        help="Dataset for reasoning mode.")

    parser.add_argument("--train_path", type=str, default=None,
                        help="Path to training data. Auto-detected if not set.")
    parser.add_argument("--valid_path", type=str, default=None,
                        help="Path to validation data. Auto-detected if not set.")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max sequence length (chat mode).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling probability.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for generation.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for chat mode (single GPU).")
    parser.add_argument("--train_device", type=str, default="cuda:0",
                        help="Training device for reasoning mode.")
    parser.add_argument("--valid_device", type=str, default="cuda:1",
                        help="Validation device for reasoning mode.")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=50)
    parser.add_argument("--val_batches", type=int, default=50,
                        help="Number of validation batches (chat mode).")

    args = parser.parse_args()

    args.wandb_project = "SFT-Reason" if args.mode == "reasoning" else "SFT4Chat"
    args.train_path = args.train_path or (f"../data/{args.dataset}/train.jsonl" if args.mode == "reasoning" else "../data/ultrachat/train.jsonl")
    args.valid_path = args.valid_path or (f"../data/{args.dataset}/valid.jsonl" if args.mode == "reasoning" else "../data/ultrachat/valid.jsonl")
    template_name = args.dataset if args.mode == "reasoning" else "rlhf"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f">>> Loading prompt template for {template_name}")
    args.prompt_template = load_prompt_template(template_name)

    device = args.device if args.mode == "chat" else args.train_device
    print(f">>> Loading model and tokenizer: {args.model_id}")
    model, tokenizer = init_policy(args.model_id, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "chat":
        print(">>> Loading training and validation data...")
        train_dataset = InstructionTuningDataset(
            tokenizer=tokenizer,
            dataset_path=args.train_path,
            seq_length=args.max_length,
            template=args.prompt_template,
            shuffle=True,
            prompt_key="prompt",
            response_key="response",
        )
        valid_dataset = InstructionTuningDataset(
            tokenizer=tokenizer,
            dataset_path=args.valid_path,
            seq_length=args.max_length,
            template=args.prompt_template,
            shuffle=False,
            prompt_key="prompt",
            response_key="response",
        )
        data_size = len(train_dataset)
    else:  # reasoning
        print(">>> Loading training and validation data...")
        train_data = load_jsonl(args.train_path)
        valid_data = load_jsonl(args.valid_path)
        data_size = len(train_data)

        reward_fn = get_reward_fn(args.dataset)

    print(">>> Initializing optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    grad_accum_steps = args.batch_size // args.micro_batch

    if args.mode == "chat":
        micro_batch_per_epoch = data_size // args.micro_batch
        training_steps = args.epochs * (micro_batch_per_epoch // grad_accum_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.05 * training_steps),
            num_training_steps=training_steps,
        )
    else:  # reasoning
        micro_batch_per_epoch = math.ceil(data_size / args.micro_batch)
        training_steps = args.epochs * (micro_batch_per_epoch // grad_accum_steps)
        warmup_steps = int(0.10 * training_steps)

        def _cosine_with_floor(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda=_cosine_with_floor)

    
    if args.mode == "chat":
        run_name = f"sft_ultrachat_lr{args.lr}"
        trainer = SFT4ChatTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            output_dir="../checkpoints/SFT4Chat",
            wandb_project=args.wandb_project,
            run_name=run_name,
        )
    else:  # reasoning
        print(">>> Initializing validation model...")
        valid_model = init_vllm(args.model_id, args.valid_device, args.seed)
        valid_sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=["</answer>", "</solution>", "```"],
            include_stop_str_in_output=True,
        )

        run_name = f"sft_{args.dataset}_lr{args.lr}"
        trainer = SFT4ReasTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            valid_data=valid_data,
            reward_fun=reward_fn,
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
