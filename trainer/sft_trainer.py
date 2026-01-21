import os
import json
import wandb
import random
import argparse
import tqdm as tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from vllm import SamplingParams

from algorithms.sft import *
from utils.vllm_helper import *
from utils.rewards import r1_zero_reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT on Qwen-Math model.")
    # --- Paths and Models ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Base model ID from Hugging Face.")
    parser.add_argument("--train_path", type=str, default="../data/math/sft_naive.jsonl", help="Path to SFT train data.")
    parser.add_argument("--valid_path", type=str, default="../data/math/test.jsonl",      help="Path to valid data.")
    parser.add_argument("--top_p", type=float, default=1.0,  help="Top-p sampling probability.")
    parser.add_argument("--temperature", type=float, default=1.0,  help="Sampling temperature.")
    parser.add_argument("--min_tokens", type=int, default=8,    help="Minimum number of tokens.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens.")

    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Total batch size.")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    # --- Logistics ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--policy_device", type=str, default="cuda:2", help="Device for the training policy.")
    parser.add_argument("--eval_device",   type=str, default="cuda:3", help="Device for the vLLM evaluation instance.")
    parser.add_argument("--wandb_project", type=str, default="SFT-MATH", help="W&B project name.")

    args = parser.parse_args()

    assert args.batch_size % args.micro_batch_size == 0, "Batch size must be divisible by micro batch size."
    args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("Loading train and valid data...")
    with open(args.train_path, 'r') as f:
        full_train_data = [json.loads(line) for line in f]
    with open(args.valid_path, 'r') as f:
        full_valid_data = [json.loads(line) for line in f]
    
    # Prepare evaluation data and vLLM instance
    problems = [item["problem"]  for item in full_valid_data]
    solution = [item["solution"] for item in full_valid_data]

    # -----------------------------------------
    #  Initialize vLLM instance for evaluation
    # -----------------------------------------
    eval_vllm = init_vllm(args.model_id, args.eval_device, args.seed)
    eval_sampling_params = SamplingParams(
        temperature=args.temperature,  # 
        top_p=args.top_p,              # 
        min_tokens=args.min_tokens,    # 
        max_tokens=args.max_tokens,    # 
        stop=["<\answer>"],            # 
        include_stop_str_in_output=True
    )

    # ==========================================
    #  PART 1: Run SFT on varying dataset sizes
    # ==========================================
    print("")
    experiment_configs = [
        # {"name": "128_examples", "data": full_train_data[:128]},
        # {"name": "256_examples", "data": full_train_data[:256]},
        # {"name": "512_examples", "data": full_train_data[:512]},
        # {"name": "1024_examples", "data": full_train_data[:1024]},
        {"name": "full_dataset", "data": full_train_data},
    ]

    for config in experiment_configs:
        train_data = config["data"]
        run_name = f"sft_{config['name']}_lr{args.lr}"
        print(f"\n> Starting SFT run: {run_name} with {len(train_data)} examples")

        wandb.init(project=args.wandb_project, name=run_name, config=args, reinit=True)
        wandb.define_metric("train_step")
        wandb.define_metric("valid_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("valid/*", step_metric="valid_step")
        
        # ---------------------------------------
        #  Initialize tokenizer and policy model
        # ---------------------------------------
        policy, tokenizer = init_policy(args.model_id, args.policy_device)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),)

        def collate_fn(batch):
            prompts   = [b['prompt']   for b in batch]
            responses = [b['response'] for b in batch]
            return tokenize_prompt_and_output(prompts, responses, tokenizer)
        train_loader = DataLoader(train_data, batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate_fn)
        
        train_step, valid_step = 0, 0
        best_acc = 0.0
        policy.train()
        
        # ======================= #
        # Start SFT Training Loop #
        # ======================= #
        for epoch in range(args.epochs):
            
            # ====== Training Logic ====== # 
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):  #
                # prepare batch data and move to device
                input_ids = batch['input_ids'].to(args.policy_device)
                labels = batch['labels'].to(args.policy_device)
                response_mask = batch['response_mask'].to(args.policy_device)
                
                results = get_response_log_probs(policy, input_ids, labels, True)
                
                # call the microbatch train step function
                loss, metadata = sft_microbatch_train_step(
                    results['log_probs'], response_mask, args.gradient_accumulation_steps,
                    normalize_constant=response_mask.sum()
                )
                
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    wandb.log({"train/loss": loss.item(), "train_step": train_step})
                    train_step += 1
                    
            # ====== Validation Logic ====== #
            policy.eval()
            load_policy_into_vllm_instance(policy, eval_vllm)
            output_path = f"./results/math/finetune.jsonl" if epoch == args.epochs - 1 else None
            acc = evaluate_vllm(eval_vllm, r1_zero_reward_fn, problems, solution, eval_sampling_params, output_path)
        
            policy.train()
            valid_step += 1
            print(f"Step {train_step}: Validation Accuracy = {acc:.2f}%")

            if acc > best_acc:  # store best model
                best_acc = acc
                model_save_path = f"../checkpoints/Qwen2.5-Math-1.5B/{run_name}_best.pt"
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(policy.state_dict(), model_save_path)
                print(f"New best model saved to {model_save_path} with accuracy {best_acc:.2f}%")
            wandb.log({"valid/accuracy": acc, "valid_step": valid_step})

            optimizer.zero_grad()

        del policy, optimizer, train_loader
        torch.cuda.empty_cache()
        wandb.finish()

    print("\nAll experiments complete.")
