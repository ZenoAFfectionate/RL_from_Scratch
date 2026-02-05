import os
import sys
import json
import time
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.rewards import r1_zero_reward_fn
from algorithms.speculative import SpeculativeDecoder

R1_ZERO_PROMPT = """
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


def run_single_mode(decoder, prompts, args):
    """Run inference in single-prompt mode with async pipeline."""
    responses = []
    total_gen_time = 0.0
    total_accepted = 0
    total_proposed = 0
    
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing prompt {i + 1}/{len(prompts)}...")
        
        gen_start = time.time()
        response = decoder.speculative_decode(
            prompt=prompt,
            max_tokens=args.max_tokens,
            num_speculative_tokens=args.num_speculative_tokens,
            use_async_pipeline=args.use_async_pipeline
        )
        total_gen_time += time.time() - gen_start
        
        # Extract only generated part (remove prompt)
        prompt_len = len(prompt)
        generated_text = response[prompt_len:] if response.startswith(prompt) else response
        responses.append(generated_text)
    
    return responses, total_gen_time


def run_batch_mode(decoder, prompts, args):
    """Run inference in continuous batching mode."""
    print(f"\nUsing Continuous Batching (batch_size={args.batch_size})...")
    
    scheduler = decoder.create_batch_scheduler(
        max_batch_size=args.batch_size,
        num_speculative_tokens=args.num_speculative_tokens
    )
    
    gen_start = time.time()
    results = scheduler.run_until_complete(
        prompts=prompts,
        max_tokens=args.max_tokens,
        verbose=True
    )
    total_gen_time = time.time() - gen_start
    
    # Extract generated text from results
    responses = [r["generated_text"] for r in results]
    
    # Collect statistics
    stats = scheduler.get_stats()
    avg_acceptance_rate = np.mean([r["acceptance_rate"] for r in results])
    
    print(f"\nBatch Processing Statistics:")
    print(f"  - Total iterations: {stats['total_iterations']}")
    print(f"  - Sequences processed: {stats['total_sequences_processed']}")
    print(f"  - Average acceptance rate: {avg_acceptance_rate:.2%}")
    
    return responses, total_gen_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate speculative decoding on a specified dataset.")

    parser.add_argument(
        "--draft_model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="The name or path of the draft model to use from Hugging Face Hub."
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="The name or path of the target model to use from Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="math",
        help="The dataset to evaluate on."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--num_speculative_tokens",
        type=int,
        default=16,
        help="Number of tokens to speculate per iteration."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Maximum batch size for continuous batching."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="batch",
        help="Inference mode: 'single' for one-by-one with async pipeline, 'batch' for continuous batching."
    )
    parser.add_argument(
        "--use_async_pipeline",
        action="store_true",
        default=True,
        help="Use async draft-verify pipeline (only for single mode)."
    )
    parser.add_argument(
        "--adaptive_k",
        action="store_true",
        default=True,
        help="Use adaptive speculation length."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)."
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Speculative Decoding Inference")
    print("=" * 60)
    print(f"  Draft Model:            {args.draft_model}")
    print(f"  Target Model:           {args.target_model}")
    print(f"  Dataset:                {args.dataset_name}")
    print(f"  Max Tokens:             {args.max_tokens}")
    print(f"  Num Speculative Tokens: {args.num_speculative_tokens}")
    print(f"  Mode:                   {args.mode}")
    if args.mode == "single":
        print(f"  Async Pipeline:         {args.use_async_pipeline}")
    else:
        print(f"  Batch Size:             {args.batch_size}")
    print(f"  Adaptive K:             {args.adaptive_k}")
    print("=" * 60)

    # Load dataset
    dataset = []
    print(f"\nLoading validation set from ../data/{args.dataset_name}/valid.jsonl ...", end=' ')
    with open(f"./data/{args.dataset_name}/valid.jsonl", "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    print(f"Done!")
    
    # Optionally limit samples
    if args.num_samples is not None:
        dataset = dataset[:args.num_samples]
    print(f"Evaluating on {len(dataset)} examples.\n")

    # Format prompts
    prompts = [R1_ZERO_PROMPT.format(question=example["problem"]) for example in dataset]
    ground_truth_answers = [example["solution"] for example in dataset]

    # Initialize SpeculativeDecoder
    print("Initializing SpeculativeDecoder...")
    decoder = SpeculativeDecoder(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        adaptive_k=args.adaptive_k
    )
    print("Decoder initialized successfully.\n")

    # Run inference based on mode
    if args.mode == "single":
        responses, total_gen_time = run_single_mode(decoder, prompts, args)
    else:
        responses, total_gen_time = run_batch_mode(decoder, prompts, args)

    # Validate generated responses using reward function
    results = []
    print("\nValidating responses...")
    val_start = time.time()
    for i, (response, ground_truth) in enumerate(zip(responses, ground_truth_answers)):
        reward_result = r1_zero_reward_fn(response, ground_truth)

        result = {
            "response": response,
            "ground_truth": ground_truth,
            "format_reward": reward_result["format_reward"],
            "answer_reward": reward_result["answer_reward"],
            "is_correct": reward_result["reward"] > 0,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"Validated {i + 1}/{len(responses)} responses")
    total_val_time = time.time() - val_start
    print(f"Validation complete. ({total_val_time:.2f}s)")

    # Save results to JSONL file
    os.makedirs("./results", exist_ok=True)
    output_file = f"./results/speculative_{args.mode}_{args.dataset_name}.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\nResults saved to {output_file}")

    # Compute summary statistics
    total_problems = len(responses)
    format_correct = sum(1 for r in results if r["format_reward"] == 1.0)
    answer_correct = sum(1 for r in results if r["is_correct"])

    # Print results summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Total Problems:   {total_problems}")
    print(f"  Format Correct:   {format_correct} ({format_correct / total_problems * 100:.2f}%)")
    print(f"  Answer Correct:   {answer_correct} ({answer_correct / total_problems * 100:.2f}%)")
    print(f"  Overall Accuracy: {answer_correct / total_problems * 100:.2f}%")
    print("=" * 60)

    # Print timing breakdown
    total_time = total_gen_time + total_val_time
    total_tokens = sum(len(r) for r in responses)  # Approximate token count
    
    print("\nTiming Breakdown")
    print("=" * 60)
    print(f"  Total time:           {total_time:.2f}s")
    print(f"  Generation time:      {total_gen_time:.2f}s ({total_gen_time / total_time * 100:.1f}%)")
    print(f"  Validation time:      {total_val_time:.2f}s ({total_val_time / total_time * 100:.1f}%)")
    print(f"  Throughput:           {total_problems / total_gen_time:.2f} prompts/sec")
    print(f"  Avg time per prompt:  {total_gen_time / total_problems:.2f}s")
    print("=" * 60)
