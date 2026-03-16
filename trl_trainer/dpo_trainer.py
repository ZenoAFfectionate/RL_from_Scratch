"""
TRL-based DPO Trainer for RLHF preference data.

Functionally equivalent to trainer/dpo_trainer.py.

Usage:
  python dpo_trainer.py --model_id Qwen/Qwen3.5-2B --dataset rlhf
  python dpo_trainer.py --model_id Qwen/Qwen3.5-2B --init_checkpoint ../checkpoints/SFT4Chat/sft_final.pt
"""

import os
import sys
import json
import random
import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import DPOTrainer, DPOConfig

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from trl_trainer.utils import load_prompt_template, FileLoggingCallback, run_post_training_eval


def build_dpo_dataset(data_path, template):
    """
    Build a HuggingFace Dataset for DPO training from RLHF preference data.

    Input JSONL format:
        {"instruction": ..., "chosen": ..., "rejected": ...}

    TRL's DPOTrainer expects columns: prompt, chosen, rejected.
    The prompt is the formatted instruction prefix; chosen/rejected are the
    full formatted texts (prompt + response).
    """
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    prompt_prefix = template.split("{response}")[0]

    prompts = []
    chosens = []
    rejecteds = []

    for item in raw_data:
        prompt = prompt_prefix.format(instruction=item["instruction"])
        chosen = template.format(instruction=item["instruction"], response=item["chosen"])
        rejected = template.format(instruction=item["instruction"], response=item["rejected"])

        prompts.append(prompt)
        chosens.append(chosen)
        rejecteds.append(rejected)

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRL-based DPO Trainer")

    # --- Model ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", type=str, default="rlhf", choices=["rlhf"])
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Path to a .pt state_dict to warm-start the policy (e.g., SFT checkpoint).")

    # --- DPO Hyperparameters ---
    parser.add_argument("--beta", type=float, default=0.2, help="DPO temperature parameter.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "kto_pair", "sppo_hard", "nca_pair"],
                        help="DPO loss variant.")

    # --- Logging & Saving ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="[TRL]DPO-RLHF")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--prompt_template_path", type=str, default=None,
                        help="Path to custom prompt template file.")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "checkpoints/[TRL]DPO")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ─── Load Template ──────────────────────────────────────────────
    template = load_prompt_template("rlhf", args.prompt_template_path)

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

    # TRL DPOTrainer creates the reference model internally as a copy of the policy,
    # which matches the original trainer's behavior (separate frozen copy).

    # ─── Build Datasets ─────────────────────────────────────────────
    train_path = os.path.join(project_root, f"data/{args.dataset}/train.jsonl")
    valid_path = os.path.join(project_root, f"data/{args.dataset}/valid.jsonl")
    print(f">>> Building DPO datasets from {train_path}")

    train_dataset = build_dpo_dataset(train_path, template)
    eval_dataset = build_dpo_dataset(valid_path, template)
    print(f"    Train: {len(train_dataset)} pairs, Eval: {len(eval_dataset)} pairs")

    # ─── Configure TRL DPOTrainer ───────────────────────────────────
    gradient_accumulation_steps = args.batch_size // args.micro_batch
    run_name = f"trl_dpo_{args.dataset}_beta{args.beta}_lr{args.lr}"

    dpo_config = DPOConfig(
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
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
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
        remove_unused_columns=False,
    )

    # ─── Initialize W&B ─────────────────────────────────────────────
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # ─── Create Trainer and Train ───────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[FileLoggingCallback(args.output_dir)],
    )

    print(">>> Starting TRL DPO training...")
    trainer.train()

    # ─── Save Final Model ───────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f">>> Final model saved to {final_dir}")

    # Save .pt state_dict for compatibility with existing evaluation code
    pt_path = os.path.join(args.output_dir, f"trl_dpo_final.pt")
    torch.save(model.state_dict(), pt_path)
    print(f">>> State dict saved to {pt_path}")

    # ─── Post-Training Evaluation (generation-based) ───────────────
    import gc

    print("\n>>> Freeing training objects for post-training evaluation...")
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # DPO trains on chat/preference data — evaluate on MMLU as a quantifiable benchmark
    mmlu_eval_path = os.path.join(project_root, "data/mmlu/valid.jsonl")
    mmlu_template = load_prompt_template("mmlu")
    eval_results_path = os.path.join(args.output_dir, "trl_dpo_mmlu_eval_results.jsonl")

    print(">>> Evaluating on MMLU benchmark...")
    metrics = run_post_training_eval(
        model_path=final_dir,
        dataset_name="mmlu",
        eval_data_path=mmlu_eval_path,
        prompt_template=mmlu_template,
        output_filepath=eval_results_path,
        seed=args.seed,
        model_id=args.model_id,
    )
    print(f"\n>>> Post-training evaluation results (MMLU):")
    print(f"    Format Accuracy:  {metrics['format_accuracy']:.2f}%")
    print(f"    Answer Accuracy:  {metrics['answer_accuracy']:.2f}%")
    print(f"    Overall Accuracy: {metrics['accuracy']:.2f}%")
