import os
import sys
import json
import argparse
from vllm import LLM, SamplingParams

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.rewards import code_reward_fn
from utils.vllm_helper import evaluate_vllm

# vllm_helper sets VLLM_ENABLE_V1_MULTIPROCESSING=0 (needed for training),
# but for inference with tensor_parallel we want native multiprocessing
# instead of Ray, so re-enable it here.
os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a VLLM model on a code-generation dataset.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="The name or path of the model to use."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="code",
        help="The dataset to evaluate on."
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens to generate.")

    args = parser.parse_args()

    print(f"Running evaluation with the following parameters:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Max Tokens: {args.max_tokens}\n")

    # load prompt template
    print(f"Loading prompt template ...", end=' ')
    with open(os.path.join(project_root, "prompts", f"{args.dataset_name}.prompt"), "r") as f:
        prompt_template = f.read()
    print("Success!\nLoaded prompt template.")

    # load dataset
    dataset = []
    print(f"Loading validation set ...", end=' ')
    with open(os.path.join(project_root, "data", args.dataset_name, "valid.jsonl"), "r") as f:
        for line in f: dataset.append(json.loads(line))
    print(f"Success!\nLoaded {len(dataset)} examples.")

    # format prompts and normalize ground-truth test cases
    prompts = [
        prompt_template.format(problem=example["problem"])
        for example in dataset
    ]
    truth_tests = [example["test"] for example in dataset]

    # load vLLM model
    vllm_model = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
        include_stop_str_in_output=True,
    )
    
    # perform zero-shot evaluation on model
    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=code_reward_fn,
        prompts=prompts,
        answers=truth_tests,
        eval_sampling_params=sampling_params,
        output_filepath=os.path.join(project_root, "results", "baseline_code.jsonl"),
    )
