import os
import sys
import json
import argparse
from vllm import LLM, SamplingParams

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.rewards import r1_zero_reward_fn
from utils.vllm_helper import evaluate_vllm


R1_ZERO_PROMPT="""
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a VLLM model on a specified dataset.")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="The name or path of the model to use from Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="math",
        help="The dataset to evaluate on."
    )
    
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")
    parser.add_argument("--min_tokens", type=int, default=128,  help="Minimum number of tokens to generate.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate.")

    args = parser.parse_args()

    print(f"Running evaluation with the following parameters:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Max Tokens: {args.max_tokens}")
    print(f"  - Min Tokens: {args.min_tokens}\n")

    # load dataset and prompt template
    dataset = []  
    print(f"Loading validation set ...", end=' ')
    with open(f"../data/{args.dataset_name}/test.jsonl", "r") as f:
        for line in f: dataset.append(json.loads(line))
    print(f"Success!\nLoaded {len(dataset)} examples.")

    # format each example into a string prompt and extract the answer
    prompts = [R1_ZERO_PROMPT.format(question=example["problem"]) for example in dataset]
    ground_truth_answers = [example["solution"] for example in dataset]

    # load vLLM model and create a sampling params object
    vllm_model = LLM(model=args.model_name, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=args.temperature,  # 
        top_p=args.top_p,              # 
        min_tokens=args.min_tokens,    # 
        max_tokens=args.max_tokens,    # 
        stop=["<\answer>"],            # 
        include_stop_str_in_output=True
    )

    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=ground_truth_answers,
        eval_sampling_params=sampling_params,
        output_filepath=f"./results/baseline_math.jsonl"
    )
