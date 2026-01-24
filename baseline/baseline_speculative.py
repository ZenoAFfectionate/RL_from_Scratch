import os
import sys
import json
import time
import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a VLLM model on a specified dataset.")

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
        default=32,
        help="Number of tokens to speculate per iteration."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for parallel generation."
    )

    args = parser.parse_args()

    print(f"Running Speculative Decoding with parameters:")
    print(f"  - Draft Model: {args.draft_model}")
    print(f"  - Target Model: {args.target_model}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Max Tokens: {args.max_tokens}")
    print(f"  - Num Speculative Tokens: {args.num_speculative_tokens}")
    print(f"  - Batch Size: {args.batch_size}")

    # Load dataset and prompt template
    dataset = []
    print(f"Loading validation set ...", end=' ')
    with open(f"../data/{args.dataset_name}/test.jsonl", "r") as f:
        for line in f: dataset.append(json.loads(line))
    print(f"Success!\nLoaded {len(dataset)} examples.\n")

    # Format each example into a string prompt and extract the answer
    prompts = [R1_ZERO_PROMPT.format(
        question=example["problem"]) for example in dataset]
    ground_truth_answers = [example["solution"] for example in dataset]

    # Initialize SpeculativeDecoder
    print("Initializing SpeculativeDecoder...")
    decoder = SpeculativeDecoder(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        max_tokens=args.max_tokens
    )

    responses = []
    total_gen_time = 0.0
    total_draft_time = 0.0
    total_verify_time = 0.0
    total_overhead_time = 0.0

    # Process prompts in batches
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    for batch_idx, batch_start in enumerate(range(0, len(prompts), args.batch_size)):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        print(
            f"\n[Batch {batch_idx + 1}/{num_batches}] Processing {len(batch_prompts)} prompts...")

        # Generate outputs using speculative decoding with all optimizations
        gen_start = time.time()
        batch_responses = decoder.speculative_decode(
            prompts=batch_prompts,
            max_tokens=args.max_tokens,
            num_speculative_tokens=args.num_speculative_tokens,
            adaptive_speculation=True,
            log_interval=20
        )
        total_gen_time += time.time() - gen_start

        # accumulate timing stats
        total_draft_time += decoder.timing_stats["draft_time"]
        total_verify_time += decoder.timing_stats["verify_time"]
        total_overhead_time += decoder.timing_stats["overhead_time"]

        responses.extend(batch_responses)

    results = []
    # validate generated responses using reward function
    print("\nValidating responses...")
    val_start = time.time()
    for i, (response, ground_truth) in enumerate(zip(responses, ground_truth_answers)):
        # comprehensive validation using r1_zero_reward_fn
        reward_result = r1_zero_reward_fn(response, ground_truth)

        result = {
            "response": response,
            "ground_truth": ground_truth,
            "format_reward": reward_result["format_reward"],
            "answer_reward": reward_result["answer_reward"],
            "is_correct": True if reward_result["reward"] > 0 else False,
        }
        results.append(result)

        if (i + 1) % 500 == 0:
            print(f"Validated {i+1}/{len(responses)} responses")
    total_val_time = time.time() - val_start

    # store the results to a JSONL file
    with open("../data/results/baseline_speculative.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # compute summary statistics
    total_problems = len(results)
    format_correct = sum(1 for r in results if r["format_reward"] > 0)
    answer_correct = sum(1 for r in results if r["is_correct"])

    # print the results summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Total Problems:   {total_problems}")
    print(f"  Format Correct:   {format_correct} ({format_correct / total_problems * 100:.2f}%)")
    print(f"  Answer Correct:   {answer_correct} ({answer_correct / total_problems * 100:.2f}%)")
    print(f"  Overall Accuracy: {answer_correct / total_problems * 100:.2f}%")
    print("=" * 60)

    # print timing breakdown
    total_time = total_gen_time + total_val_time
    print("\nTiming Breakdown")
    print("=" * 60)
    print(f"  Total time:       {total_time:.2f}s")
    print(f"  Generation time:  {total_gen_time:.2f}s ({total_gen_time/total_time*100:.1f}%)")
    print(f"   - Draft:         {total_draft_time:.2f}s ({total_draft_time/total_time*100:.1f}%)")
    print(f"   - Verify:        {total_verify_time:.2f}s ({total_verify_time/total_time*100:.1f}%)")
    print(f"   - Overhead:      {total_overhead_time:.2f}s ({total_overhead_time/total_time*100:.1f}%)")
    print(f"  Validation time:  {total_val_time:.2f}s ({total_val_time/total_time*100:.1f}%)")
    print("=" * 60)
