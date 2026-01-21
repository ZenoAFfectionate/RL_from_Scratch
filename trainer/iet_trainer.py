import os
import json
import wandb
import argparse
import tqdm as tqdm
import random
import numpy as np
from typing import List, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from vllm import SamplingParams

from algorithms.sft import *
from utils.vllm_helper import *
from utils.rewards import r1_zero_reward_fn


def sample_batch_from_dataset(dataset: List[Dict], batch_size: int) -> List[Dict]:
    """Sample a random batch of questions from the dataset."""
    return random.sample(dataset, min(batch_size, len(dataset)))


def generate_responses(
    vllm_model: LLM,
    questions: List[str],
    prompt_template: str,
    sampling_params: SamplingParams,
) -> List[List[str]]:
    """Generate G responses for each question using vLLM."""
    prompts = []
    for question in questions: # Generate G responses per question
        prompts.append(prompt_template.format(question=question))

    # Generate all responses at once
    request_outputs = vllm_model.generate(prompts, sampling_params)

    all_responses = []
    for output in request_outputs:
        # The 'output.outputs' attribute is a list of 'n' completions.
        question_responses = [completion.text.strip() for completion in output.outputs]
        all_responses.append(question_responses)

    return all_responses


def filter_correct_responses(
    questions: List[str],
    responses: List[List[str]],
    ground_truths: List[str],
    reward_fn
) -> List[Dict]:
    """Filter responses to keep only correct ones and format for SFT."""
    sft_data = []

    for question, question_responses, ground_truth in zip(questions, responses, ground_truths):
        for response in question_responses:
            # compute reward for this response
            reward_dict = reward_fn(response, ground_truth)

            # only keep responses with reward = 1 (correct)
            if reward_dict.get("reward", 0) == 1:
                sft_data.append({
                    "prompt": question,
                    "response": response
                })

    return sft_data


def compute_response_entropy(
    policy,
    tokenizer,
    responses: List[str],
    device: str
) -> float:
    """Compute average entropy of model responses."""
    if not responses:
        return 0.0

    # Tokenize responses
    tokenized = tokenizer(
        responses,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=2048
    )

    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    with torch.no_grad():
        outputs = policy(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute entropy using the helper function
        entropy = compute_entropy(logits)  # Shape: (batch_size, seq_len)

        # Mask out padding tokens and compute average
        entropy_masked = entropy * attention_mask.float()
        total_entropy = entropy_masked.sum()
        total_tokens  = attention_mask.sum()

        avg_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0

    return avg_entropy.item()


def run_sft(
    policy,
    tokenizer,
    sft_data: List[Dict],
    optimizer,
    epochs: int,
    batch_size: int,
    micro_batch_size: int,
    clip_grad_norm: float,
    device: str
) -> float:
    """Run SFT on filtered data and return final loss."""
    if not sft_data:
        return 0.0

    def collate_fn(batch):
        prompts = [b['prompt'] for b in batch]
        responses = [b['response'] for b in batch]
        return tokenize_prompt_and_output(prompts, responses, tokenizer)

    train_loader = DataLoader(sft_data, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn)
    gradient_accumulation_steps = batch_size // micro_batch_size

    policy.train()
    total_loss = 0.0
    num_steps = 0

    for _ in range(epochs):
        for i, batch in enumerate(train_loader):
            # Prepare batch data
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            response_mask = batch['response_mask'].to(device)

            results = get_response_log_probs(policy, input_ids, labels, True)

            # SFT training step
            loss, _ = sft_microbatch_train_step(
                results['log_probs'], response_mask, gradient_accumulation_steps,
                normalize_constant=response_mask.sum()
            )

            total_loss += loss.item()
            num_steps += 1

            if (i + 1) % gradient_accumulation_steps == 0:
                clip_grad_norm_(policy.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

    return total_loss / num_steps if num_steps > 0 else 0.0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Expert Iteration on Qwen-Math model.")

    # Model and data paths
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B",     help="Base model ID")
    parser.add_argument("--train_path", type=str, default="../data/math/train.jsonl", help="Path to MATH train data")
    parser.add_argument("--valid_path", type=str, default="../data/math/test.jsonl",  help="Path to validation data")

    # Sampling parameters
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--min_tokens", type=int, default=4,    help="Minimum tokens")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens")

    # Expert Iteration parameters
    parser.add_argument("--n_ei_steps", type=int, default=4, help="Number of expert iteration steps")
    parser.add_argument("--G", type=int, default=4, help="Number of rollouts per question")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for each EI step (size of D_b)")

    # SFT parameters
    parser.add_argument("--sft_epochs", type=int, default=4, help="Number of SFT epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping")

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N EI steps")
    parser.add_argument("--eval_set_size", type=int, default=256, help="Validation set size")

    # Logistics
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--policy_device", type=str, default="cuda:0", help="Policy device")
    parser.add_argument("--eval_device", type=str, default="cuda:1", help="Evaluation device")
    parser.add_argument("--wandb_project", type=str, default="ExpertIteration-MATH", help="W&B project")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading datasets... ", end='')
    with open(args.train_path, 'r') as f:
        full_train_data = [json.loads(line) for line in f]
    with open(args.valid_path, 'r') as f:
        full_valid_data = [json.loads(line) for line in f]
    print("Success!")

    # Load prompt template
    with open("./prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    # Prepare evaluation data
    eval_problems  = [item["problem"]  for item in full_valid_data[:args.eval_set_size]]
    eval_solutions = [item["solution"] for item in full_valid_data[:args.eval_set_size]]

    # Configure vLLM sampling to terminate at second </answer>
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        n=args.G,            # Generate G completions per prompt
        stop=["</answer>"],  # Stop at second </answer> tag
        include_stop_str_in_output=True
    )

    # Run single experiment with command line arguments
    run_name = f"expi_G{args.G}_batch{args.batch_size}_epochs{args.sft_epochs}_lr{args.lr}"
    print(f"\n> Starting Expert Iteration: {run_name}")

    # Initialize wandb
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args), reinit=True)
    wandb.define_metric("ei_step")
    wandb.define_metric("train/*", step_metric="ei_step")
    wandb.define_metric("valid/*", step_metric="ei_step")

    # Initialize models
    policy, tokenizer = init_policy(args.model_id, args.policy_device)
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = AdamW(policy.parameters(), lr=args.lr)

    # Initialize evaluation vLLM
    eval_vllm = init_vllm(args.model_id, args.eval_device, args.seed)

    # Expert Iteration main loop
    for ei_step in range(args.n_ei_steps):
        print(f"\n--- Expert Iteration Step {ei_step + 1}/{args.n_ei_steps} ---")

        # Step 1: Sample batch of questions
        batch_data = sample_batch_from_dataset(full_train_data, args.batch_size)
        questions = [item["problem"] for item in batch_data]
        ground_truths = [item["solution"] for item in batch_data]

        print(f"Sampled {len(questions)} questions")

        # Step 2: Load current policy into vLLM for generation
        policy.eval()
        load_policy_into_vllm_instance(policy, eval_vllm)

        # Step 3: Generate G responses per question
        print(f"Generating {args.G} responses per question...")
        responses = generate_responses(
            eval_vllm, questions, prompt_template, sampling_params, args.G
        )

        # Step 4: Filter correct responses
        print("Filtering correct responses...")
        sft_data = filter_correct_responses(questions, responses, ground_truths, r1_zero_reward_fn)
        print(f"Found {len(sft_data)} correct responses for SFT")

        # Step 5: Compute entropy of responses (for logging)
        all_responses = [resp for resp_list in responses for resp in resp_list]
        avg_entropy = compute_response_entropy(policy, tokenizer, all_responses, args.policy_device)

        # Step 6: Run SFT if we have any correct data
        sft_loss = 0.0
        if sft_data:
            print(f"Running SFT for {args.sft_epochs} epochs...")
            sft_loss = run_sft(
                policy, tokenizer, sft_data, optimizer,
                args.sft_epochs, args.batch_size,
                args.micro_batch_size, args.clip_grad_norm, args.policy_device
            )

        # Step 7: Evaluation
        if ei_step % args.eval_interval == 0 or ei_step == args.n_ei_steps - 1:
            print("Running evaluation...")
            policy.eval()
            load_policy_into_vllm_instance(policy, eval_vllm)

            eval_sampling_params = SamplingParams(
                temperature=0.0,  # Greedy
                top_p=1.0,        # No top-p
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                stop=["</answer>"],
                include_stop_str_in_output=True
            )

            accuracy = evaluate_vllm(eval_vllm, r1_zero_reward_fn, eval_problems, eval_solutions, eval_sampling_params)
            print(f"EI Step {ei_step + 1}: Accuracy = {accuracy:.2f}%")
        else:
            accuracy = 0.0

        wandb.log({  # Log metrics
            "ei_step": ei_step + 1,
            "train/sft_loss": sft_loss,
            "train/num_correct_responses": len(sft_data),
            "train/response_entropy": avg_entropy,
            "valid/accuracy": accuracy,
        })

    # save final model
    model_save_path = f"../checkpoints/{args.model_id}/{run_name}_final.pt"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(policy.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    # Cleanup
    del policy, optimizer, eval_vllm
    torch.cuda.empty_cache()
    wandb.finish()

    print("\nExpert Iteration complete!")