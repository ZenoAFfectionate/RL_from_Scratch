import os
import sys
import json
import argparse
from vllm import LLM, SamplingParams

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.rewards import dsr1_reward_fn
from utils.vllm_helper import evaluate_vllm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a VLLM model on a specified dataset.")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="The name or path of the model to use from Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="math",
        help="The dataset to evaluate on."
    )
    
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate.")

    args = parser.parse_args()

    print(f"Running evaluation with the following parameters:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Max Tokens: {args.max_tokens}\n")

    # load dataset and prompt template
    print(f"Loading prompt template ...", end=' ')
    with open(f"../trainer/prompts/{args.dataset_name}.prompt", "r") as f:
        prompt_template = f.read()
    print("Success!\nLoaded prompt template.")

    dataset = []
    print(f"Loading validation set ...", end=' ')
    with open(f"../data/{args.dataset_name}/train.jsonl", "r") as f:
        for line in f: dataset.append(json.loads(line))
    print(f"Success!\nLoaded {len(dataset)} examples.")

    # format each example into a string prompt and extract the answer
    prompts = [prompt_template.format(problem=example["problem"]) for example in dataset]
    ground_truth_answers = [example["solution"] for example in dataset]

    # load vLLM model and create a sampling params object
    vllm_model = LLM(
        model=args.model_name, 
        trust_remote_code=True, 
        max_model_len=args.max_tokens
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=dsr1_reward_fn,
        prompts=prompts,
        answers=ground_truth_answers,
        eval_sampling_params=sampling_params,
        output_filepath=f"../results/baseline_math.jsonl"
    )
