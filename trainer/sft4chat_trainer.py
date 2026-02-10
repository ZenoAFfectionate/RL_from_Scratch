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

from algorithms.sft4chat import SFT4ChatTrainer
from data.lm_dataset import InstructionTuningDataset
from utils.vllm_helper import init_policy

# Full Alpaca-style template for instruction tuning (includes both {instruction} and {response})
INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT on ultrachat dataset.")
    # --- Paths and Models ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_path", type=str, default="../data/ultrachat/train.jsonl")
    parser.add_argument("--valid_path", type=str, default="../data/ultrachat/valid.jsonl")
    parser.add_argument("--prompt_template_path", type=str, default="./prompts/alpaca_sft.prompt")

    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch", type=int, default=4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    # --- Logistics ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=10,  help="")
    parser.add_argument("--val_interval", type=int, default=200, help="")
    parser.add_argument("--val_batches",  type=int, default=100, help="")
    parser.add_argument("--wandb_project", type=str, default="SFT4Chat")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --------------------------------
    #  Initialize tokenizer and model
    # --------------------------------
    print(f">>> Loading model and tokenizer: {args.model_id}")
    model, tokenizer = init_policy(args.model_id, args.device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------
    #  Building dataset and prompt template
    # -------------------------------------
    print(f">>> Loading prompt template from {args.prompt_template_path}")
    with open(args.prompt_template_path, "r") as f:
        args.prompt_template = f.read()

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


    print("Initializing optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=0.0, betas=(0.9, 0.95), fused=True)

    grad_accum_steps = args.batch_size // args.micro_batch
    micro_batch_per_epoch = len(train_dataset) // args.micro_batch
    training_steps = args.epochs * (micro_batch_per_epoch // grad_accum_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * training_steps),
        num_training_steps=training_steps,
    )

    # ============================== #
    #  Create Trainer and Fine-Tune  #
    # ============================== #
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
    trainer.train()

    # =============================================== #
    #  Validate Fine-Tuned Model on Downstream Tasks  #
    # =============================================== #
