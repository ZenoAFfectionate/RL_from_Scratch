import os
import sys
import wandb
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from data.lm_dataset import RLHFDataset
from algorithms.dpo import dpo_batch_loss, evaluate_dpo


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


def collate_fn(batch):
    """Collate function that returns lists of strings."""
    instructions = [item['instruction'] for item in batch]
    chosen = [item['chosen'] for item in batch]
    rejected = [item['rejected'] for item in batch]
    return {
        'instructions': instructions,
        'chosen': chosen,
        'rejected': rejected,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO training on RLHF preference data.")

    # --- Paths and Models ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_path", type=str, default="../data/rlhf/train.jsonl")
    parser.add_argument("--valid_path", type=str, default="../data/rlhf/valid.jsonl")
    
    # --- DPO Specific Hyperparameters ---
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Total batch size.")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro batch for gradient accumulation.")
    
    # --- Training / Optimization ---
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    
    # --- Logistics ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--policy_device", type=str, default="cuda:0", help="Device for the training policy.")
    parser.add_argument("--ref_device", type=str, default="cuda:1", help="Device for the reference model.")
    parser.add_argument("--wandb_project", type=str, default="DPO-RLHF", help="W&B project name.")
    parser.add_argument("--save_dir", type=str, default="../checkpoints/dpo", help="Directory to save checkpoints.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps.")

    args = parser.parse_args()

    # Validate arguments
    assert args.batch_size % args.micro_batch_size == 0, "Batch size must be divisible by micro batch size."
    args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Calculated gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    # Initialize wandb
    run_name = f"dpo_beta{args.beta}_lr{args.lr}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # ============================ #
    # Load training and validation data
    # ============================ #
    print(f"\n>>> Loading training data from: {args.train_path}")
    train_dataset = RLHFDataset(args.train_path)
    print(f"    Loaded {len(train_dataset)} training examples")
    
    print(f">>> Loading validation data from: {args.valid_path}")
    valid_dataset = RLHFDataset(args.valid_path)
    print(f"    Loaded {len(valid_dataset)} validation examples")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.micro_batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.micro_batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )

    # =============================== #
    # Initialize models and tokenizer
    # =============================== #
    print(f"\n>>> Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f">>> Loading policy model to {args.policy_device}...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    policy_model.to(args.policy_device)

    print(f">>> Loading reference model to {args.ref_device}...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    reference_model.to(args.ref_device)
    reference_model.eval()  # Reference model is always in eval mode
    # freeze reference model
    for param in reference_model.parameters(): 
        param.requires_grad = False

    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    # Calculate total steps for learning rate scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    print(f"\n{'='*60}")
    print(f"Starting DPO Training")
    print(f"{'='*60}")
    print(f"  Model: {args.model_id}")
    print(f"  Beta: {args.beta}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size} (micro: {args.micro_batch_size})")
    print(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Total Training Steps: {total_steps}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Training Examples: {len(train_dataset)}")
    print(f"  Validation Examples: {len(valid_dataset)}")
    print(f"{'='*60}\n")

    # ================== #
    # DPO Training Loop  #
    # ================== #
    global_step = 0
    accumu_step = 0
    best_val_accuracy = 0.0
    
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs}")
        
        policy_model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Compute DPO loss
            loss, metadata = dpo_batch_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                tokenizer=tokenizer,
                instructions=batch['instructions'],
                chosen_responses=batch['chosen'],
                rejected_responses=batch['rejected'],
                template=ALPACA_TEMPLATE,
                beta=args.beta,
            )
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            
            epoch_loss += metadata['loss']
            epoch_accuracy += metadata['accuracy']
            num_batches += 1
            accumu_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{metadata['loss']:.4f}",
                "acc": f"{metadata['accuracy']:.2%}"
            })
            
            # Perform optimizer step when accumulation is done
            if accumu_step % args.gradient_accumulation_steps == 0:
                grad_norm = clip_grad_norm_(policy_model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                # Log to wandb
                wandb.log({
                    "train/loss": metadata['loss'],
                    "train/accuracy": metadata['accuracy'],
                    "train/reward_margin_mean": metadata['reward_margin_mean'],
                    "train/chosen_log_ratio": metadata['chosen_log_ratio_mean'],
                    "train/rejected_log_ratio": metadata['rejected_log_ratio_mean'],
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "step": global_step,
                })
                
                global_step += 1
                
                # Evaluate periodically
                if global_step % args.eval_steps == 0:
                    print(f"\nStep {global_step}: Running Validation...")
                    val_metrics = evaluate_dpo(
                        policy_model, reference_model, tokenizer,
                        valid_loader, ALPACA_TEMPLATE, args.beta
                    )
                    print(f"  Validation Loss: {val_metrics['loss']:.4f}")
                    print(f"  Validation Accuracy: {val_metrics['accuracy']:.2%}")
                    
                    wandb.log({
                        "valid/loss": val_metrics['loss'],
                        "valid/accuracy": val_metrics['accuracy'],
                        "step": global_step,
                    })
                    
                    if val_metrics['accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_metrics['accuracy']
                        save_path = os.path.join(args.save_dir, "best_model.pt")
                        torch.save(policy_model.state_dict(), save_path)
                        print(f"  New best model saved! (accuracy: {best_val_accuracy:.2%})")
                    
                    policy_model.train()
                
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.save_dir, f"checkpoint_step{global_step}.pt")
                    torch.save({
                        'step': global_step,
                        'model_state_dict': policy_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
                    print(f"\nCheckpoint saved to {save_path}")
        
        # End of epoch summary
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.2%}")

    # Final evaluation
    print("\n>>> Final Validation...")
    val_metrics = evaluate_dpo(
        policy_model, reference_model, tokenizer,
        valid_loader, ALPACA_TEMPLATE, args.beta
    )
    print(f"Final Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Final Validation Accuracy: {val_metrics['accuracy']:.2%}")

    # Save final model
    final_save_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(policy_model.state_dict(), final_save_path)
    print(f"\nFinal model saved to {final_save_path}")

    print("\n>>> Training Complete.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2%}")
    wandb.finish()
