import os
import sys
import json
import random
import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTTrainer, SFTConfig

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from trl_trainer.utils import load_prompt_template, get_base_reward_fn, FileLoggingCallback


def build_chat_dataset(data_path, prompt_template):
    """
    Build a HuggingFace Dataset for chat SFT (ultrachat).
    The template uses {instruction} and {response} placeholders.
    Data format: {"prompt": ..., "response": ...}
    """
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    texts = []
    for item in raw_data:
        text = prompt_template.format(
            instruction=item["prompt"],
            response=item["response"],
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def build_reasoning_dataset(data_path, prompt_template, dataset_name):
    """
    Build a HuggingFace Dataset for reasoning SFT (math/code/gsmk).
    Data format: {"problem": ..., "solution": ...} (or {"question": ...} for gsmk)
    Uses completion-only loss: only the solution tokens are trained on.
    """
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    texts = []
    for item in raw_data:
        if dataset_name == "gsmk":
            prompt = prompt_template.format(question=item["problem"])
        else:
            prompt = prompt_template.format(problem=item["problem"])
        # For reasoning datasets, the full training text is prompt + solution
        full_text = prompt + item["solution"]
        texts.append(full_text)

    return Dataset.from_dict({"text": texts})


def get_reward_fn(dataset_name):
    """Return the appropriate reward function for the dataset."""
    return get_base_reward_fn(dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRL-based SFT Trainer")

    # --- Mode Selection ---
    parser.add_argument("--mode", type=str, default="chat",
                        choices=["chat", "reasoning"],
                        help="Training mode: 'chat' for ultrachat, 'reasoning' for math/code/gsmk.")
    parser.add_argument("--dataset", type=str, default="math",
                        choices=["math", "gsmk", "code"],
                        help="Dataset for reasoning mode.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")

    # --- Data Paths ---
    parser.add_argument("--train_path", type=str, default="data/ultrachat/train.jsonl",
                        help="Path to training data. Auto-detected if not set.")
    parser.add_argument("--valid_path", type=str, default="data/ultrachat/valid.jsonl",
                        help="Path to validate data. Auto-detected if not set.")
    parser.add_argument("--prompt_template_path", type=str, default=None,
                        help="Path to custom prompt template file.")

    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--packing", action="store_true", default=False,
                        help="Enable sequence packing for efficiency (chat mode).")

    # --- Logging & Saving ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    if args.mode != "chat": args.train_path = f"data/{args.dataset}/train.jsonl"
    if args.mode != "chat": args.valid_path = f"data/{args.dataset}/valid.jsonl"
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "checkpoints/[TRL]SFT4Chat" if args.mode == "chat" else f"checkpoints/[TRL]SFT4{args.dataset}")
    if args.wandb_project is None:
        args.wandb_project = "[TRL]SFT4Chat" if args.mode == "chat" else f"[TRL]SFT4{args.dataset}"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f">>> Loading model and tokenizer: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # buliding datasets
    template_name = "rlhf" if args.mode == "chat" else args.dataset
    prompt_template = load_prompt_template(template_name, args.prompt_template_path)
    print(f">>> Building datasets from {args.train_path}")

    if args.mode == "chat":
        train_dataset = build_chat_dataset(args.train_path, prompt_template)
        eval_dataset = build_chat_dataset(args.valid_path, prompt_template)
    else:
        train_dataset = build_reasoning_dataset(args.train_path, prompt_template, args.dataset)
        eval_dataset = build_reasoning_dataset(args.valid_path, prompt_template, args.dataset)

    print(f"    Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")

    # configuration for SFT Trainer
    gradient_accumulation_steps = args.batch_size // args.micro_batch

    run_name = (f"trl_sft_ultrachat_lr{args.lr}" if args.mode == "chat"
                else f"trl_sft_{args.dataset}_lr{args.lr}")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=args.clip_grad_norm,
        max_length=args.max_length,
        packing=args.packing if args.mode == "chat" else False,
        dataset_text_field="text",
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
        gradient_checkpointing=True if args.mode == "reasoning" else False,
    )

    # initialize wandb project
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # create Trainer and perform training
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[FileLoggingCallback(args.output_dir)],
    )

    print(">>> Starting TRL SFT training...")
    trainer.train()

    # save trained model and tokenizer to ckpt
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f">>> Final model saved to {final_dir}")

    pt_path = os.path.join(args.output_dir, f"{run_name}_final.pt")
    torch.save(model.state_dict(), pt_path)
    print(f">>> State dict saved to {pt_path}")
