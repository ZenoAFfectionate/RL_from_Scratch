import os
import sys
import json
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from algorithms.speculative import SpeculativeDecoder
from utils.rewards import r1_zero_reward_fn


R1_ZERO_PROMPT="""
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a VLLM model on a specified dataset.")
    
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
        "--device",
        type=str,
        default="cuda",
        help="Device to run models on (cuda or cpu)."
    )

    args = parser.parse_args()

    print(f"Running Speculative Decoding with parameters:")
    print(f"  - Draft Model: {args.draft_model}")
    print(f"  - Target Model: {args.target_model}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Max Tokens: {args.max_tokens}")
    print(f"  - Num Speculative Tokens: {args.num_speculative_tokens}")
    print(f"  - Device: {args.device}\n")

    # Load dataset and prompt template
    dataset = []
    print(f"Loading validation set ...", end=' ')
    with open(f"./data/{args.dataset_name}/test.jsonl", "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    print(f"Success!\nLoaded {len(dataset)} examples.\n")

    # Format each example into a string prompt and extract the answer
    prompts = [R1_ZERO_PROMPT.format(question=example["problem"]) for example in dataset]
    ground_truth_answers = [example["solution"] for example in dataset]

    # Initialize SpeculativeDecoder
    print("Initializing SpeculativeDecoder...")
    decoder = SpeculativeDecoder(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        device=args.device
    )

    responses = []
    # generate responses for each prompt
    for i, prompt in enumerate(prompts):
        print(f"Solving Problem {i+1}/{len(prompts)}:")

        # Generate output using speculative decoding
        response = decoder.speculative_decode(
            prompt=prompt,
            max_tokens=args.max_tokens,
            num_speculative_tokens=args.num_speculative_tokens
        )
        responses.append(response)

    # validate generated responses using reward function
    for response, ground_truth in zip(responses, ground_truth_answers):
        # comprehensive validation using r1_zero_reward_fn
        reward_result = r1_zero_reward_fn(response, ground_truth)

        result = {
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "format_reward": reward_result["format_reward"],
            "answer_reward": reward_result["answer_reward"],
            "is_correct": True if reward_result["reward"] > 0 else False,
        }
        
        
    