import json
import wandb
import random
import argparse
import numpy as np
import tqdm as tqdm

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from algorithms.sft import *
from algorithms.grpo import *
from utils.vllm_helper import *
from utils.rewards import r1_zero_reward_fn
from utils.math_utils import build_train_log_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO on Qwen-Math model.")

    # --- Paths and Models ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Base model ID.")
    parser.add_argument("--train_path", type=str, default="../data/math/train.jsonl", help="Path to train data.")
    parser.add_argument("--valid_path", type=str, default="../data/math/test.jsonl", help="Path to valid data.")
    
    # --- GRPO Specific Hyperparameters ---
    parser.add_argument("--n_grpo_steps", type=int, default=200, help="Number of GRPO steps.")
    parser.add_argument("--rollout_batch_size", type=int, default=256, help="Total number of items in a rollout batch.")
    parser.add_argument("--group_size", type=int, default=8, help="Number of generations per prompt.")
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1, help="Number of training epochs per rollout.")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Training batch size (must match rollout for on-policy).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128, help="Steps to accumulate gradients.")
    parser.add_argument("--advantage_eps", type=float, default=1e-6, help="Epsilon for advantage normalization.")
    
    # --- Sampling / Generation Params ---
    parser.add_argument("--sampling_temperature", type=float, default=1.0, help="Temperature for rollout generation.")
    parser.add_argument("--sampling_min_tokens", type=int, default=4, help="Min tokens for generation.")
    parser.add_argument("--sampling_max_tokens", type=int, default=1024, help="Max tokens for generation.")
    
    # --- Training / Optimization ---
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory utilization.")
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline", 
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
                        help="Type of loss function to use.")
    parser.add_argument("--use_std_normalization", action="store_true", default=True, help="Whether to normalize advantages by std.")
    
    # --- Logistics ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--policy_device", type=str, default="cuda:0", help="Device for the training policy.")
    parser.add_argument("--eval_device", type=str, default="cuda:1", help="Device for the vLLM instance.")
    parser.add_argument("--wandb_project", type=str, default="GRPO-MATH", help="W&B project name.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps")
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    assert args.rollout_batch_size % args.group_size == 0, (
        "rollout_batch_size must be divisible by group_size")
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size

    assert args.train_batch_size >= args.group_size, (
        "train_batch_size must be greater than or equal to group_size")
    
    print(f"Calculated micro_batch_size: {micro_train_batch_size}")
    print(f"Prompts per rollout: {n_prompts_per_rollout_batch}")

    wandb.init(project=args.wandb_project, config=vars(args))


    # ============================ #
    # Loading data prompt template #
    # ============================ #
    print(f">>> Loading R1-zero prompt template...")
    with open("./prompts/r1_zero.prompt", "r") as f:
        R1_ZERO_PROMPT = f.read()
    print(">>> Loading training and validation data...")
    with open(args.train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(args.valid_path, 'r') as f:
        valid_data = [json.loads(line) for line in f]

    # =============================== #
    # Initialize models by using vLLM #
    # =============================== #
    print(">>> Initializing Policy Model...")
    policy, tokenizer = init_policy(args.model_id, args.policy_device)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(">>> Initializing vLLM Model...")
    vllm_model = init_vllm(args.model_id, args.eval_device, args.seed, args.gpu_memory_utilization)

    rollout_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        min_tokens=args.sampling_min_tokens,
        max_tokens=args.sampling_max_tokens,
        stop=["<|endoftext|>", "</s>", "\n\n\n"], 
        n=args.group_size, # Generate N responses per prompt
    )

    eval_sampling_params = SamplingParams(
        temperature=0.0,
        min_tokens=args.sampling_min_tokens,
        max_tokens=args.sampling_max_tokens,
        stop=["<|endoftext|>", "</s>"]
    )

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))


    # ================== #
    # GRPO Training Loop #
    # ================== #
    print("\n>>> Starting GRPO Training Loop")
    global_step = 0
    accumu_step = 0
    for step in tqdm(range(args.n_grpo_steps), desc="GRPO Steps"):
        # --------------------
        # 1. generate rollouts
        # --------------------
        load_policy_into_vllm_instance(policy, vllm_model)

        batch_indices = np.random.choice(len(train_data), n_prompts_per_rollout_batch, replace=False)
        batch_items = [train_data[i] for i in batch_indices]

        prompts = [R1_ZERO_PROMPT.format(question=item["problem"]) for item in batch_items]
        grounds = [item["solution"] for item in batch_items]

        rollout_outputs = vllm_model.generate(prompts, rollout_sampling_params, use_tqdm=False)

        # ---------------------------------
        # 2. compute rewards and advantages
        # ---------------------------------
        flat_prompts, flat_responses, repeated_grounds = [], [], []
        for i, output in enumerate(rollout_outputs):
            prompt = prompts[i]
            ground = grounds[i]
            for sample in output.outputs:
                flat_prompts.append(prompt)
                flat_responses.append(sample.text)
                repeated_grounds.append(ground)
        # compute advantages and move to policy device
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=flat_responses,
            repeated_ground_truths=repeated_grounds,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization
        )
        advantages = advantages.to(args.policy_device)

        # -------------------------------
        # 3. update policy model via GRPO
        # -------------------------------
        tokenized_batch = tokenize_prompt_and_output(flat_prompts, flat_responses, tokenizer)
        # create tensor dataset and dataloader
        input_ids = tokenized_batch["input_ids"].to(args.policy_device)
        labels = tokenized_batch["labels"].to(args.policy_device)
        response_mask = tokenized_batch["response_mask"].to(args.policy_device)

        # In off-policy: pre-compute old_log_prob for GRPO-Clip
        old_log_probs = None
        if args.loss_type == "grpo_clip":
            with torch.no_grad():
                policy.eval()
                outputs = get_response_log_probs(
                    policy, input_ids, labels, return_token_entropy=False
                )
                old_log_probs = outputs["log_probs"].detach()

        # initalize tensor dataset and then dataloader
        if old_log_probs is not None:
            train_dataset = TensorDataset(input_ids, labels, response_mask, advantages, old_log_probs)
        else:
            train_dataset = TensorDataset(input_ids, labels, response_mask, advantages)
        
        train_loader = DataLoader(train_dataset, batch_size=args.micro_batch_size, shuffle=True)

        step_loss, step_entropy = .0, .0
        step_metrics = {"loss": [], "token_entropy": [], "clip_fraction": []}
        
        policy.train()
        for epoch in range(args.epochs_per_rollout_batch):
            for batch in train_loader:
                # get batch data and move to device
                if old_log_probs is not None:
                    input_ids, labels, mask, adv, old_log_probs = [t.to(args.policy_device) for t in batch]
                else: 
                    input_ids, labels, mask, adv = [t.to(args.policy_device) for t in batch]

                # get log probabilities and entropy
                model_outputs = get_response_log_probs(
                    policy,
                    input_ids,
                    labels,
                    return_token_entropy=True
                )

                # get training loss and meta info
                loss, train_meta = grpo_microbatch_train_step(
                    policy_log_probs=model_outputs["log_probs"],
                    response_mask=mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    advantages=adv,
                    old_log_probs=old_log_probs,
                    cliprange=args.cliprange,
                )

                step_metrics["loss"].append(loss.item() * args.gradient_accumulation_steps)

                if "token_entropy" in model_outputs:
                    step_metrics["token_entropy"].append(model_outputs["token_entropy"].mean().item())
            
                if "clip_fraction" in train_meta:
                    step_metrics["clip_fraction"].append(train_meta["clip_fraction"].item())

                accumu_step += 1
                # perform optimizer step when accumulation is done
                if accumu_step % args.gradient_accumulation_steps == 0:
                    grad_norm = clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                    log_dict = build_train_log_dict(
                        step_metrics,
                        reward_meta,
                        global_step,
                        grad_norm,
                        advantages.mean().item(),
                        args.loss_type
                    )
                    wandb.log(log_dict)

                    # reset metrics and update global step
                    step_metrics = {k: [] for k in step_metrics}
                    global_step += 1

            # 4. evalulate and plot metrics
            if global_step % 10 == 0:
                print(f"Step {global_step}: Running Validation...")
                load_policy_into_vllm_instance(policy, vllm_model)

                # prepare validation data
                prompts   = [R1_ZERO_PROMPT.format(question=item["problem"]) for item in valid_data]
                solutions = [item["solution"] for item in valid_data]

                # evaluate accuracy
                acc = evaluate_vllm(
                    vllm_model,
                    r1_zero_reward_fn,
                    prompts,
                    solutions,
                    eval_sampling_params
                )

                print(f"Step {global_step}: Validation Accuracy = {acc:.2f}%")
                wandb.log({"valid/accuracy": acc, "step": global_step})

    print(">>> Training Complete.")
    wandb.finish()
