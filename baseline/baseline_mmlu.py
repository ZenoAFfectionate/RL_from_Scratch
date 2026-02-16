import os
import sys
import json
import argparse

from vllm import LLM, SamplingParams

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.rewards import mmlu_reward_fn
from utils.vllm_helper import evaluate_vllm


def format_mmlu_prompt(example, prompt_template):
    """
    Constructs the prompt using ONLY the user template.
    """
    question = example["question"]
    subject = example.get("subject", "general knowledge")
    options = example["choices"]
    full_prompt = prompt_template.format(
        subject=subject,
        question=question,
        options=options
    )
    return full_prompt


def index_to_letter(index: int) -> str:
    """Convert an index (0, 1, 2, 3) to a letter (A, B, C, D)."""
    return chr(ord('A') + int(index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a VLLM model on MMLU (Zero-shot).")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_name", type=str, default="mmlu")

    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate.")

    args = parser.parse_args()

    print(f"Running MMLU Zero-shot evaluation with parameters:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Temperature: {args.temperature} (Greedy)")
    print(f"  - Top-p: {args.top_p}")


    # load prompt template
    print(f"Loading prompt template ...", end=' ')
    with open(f"../trainer/prompts/{args.dataset_name}.prompt", "r") as f:
        prompt_template = f.read()
    print("Success!\nLoaded prompt template.")

    # loading validation dataset
    dataset = []
    print(f"Loading validation set ...", end=' ')
    with open(f"../data/{args.dataset_name}/valid.jsonl", "r") as f:
        for line in f: dataset.append(json.loads(line))
    print(f"Success!\nLoaded {len(dataset)} examples.")

    # create prompts and extract ground truth from dataset
    prompts = [format_mmlu_prompt(example, prompt_template) for example in dataset]
    ground_truth = [index_to_letter(e["answer"]) for e in dataset]

    # initialize model by means of vLLM
    print(f"Initializing vLLM with model: {args.model_name}")
    vllm_model = LLM(
        model=args.model_name, 
        trust_remote_code=True, 
        max_model_len=32768
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["\n", "Question:"], 
        include_stop_str_in_output=False
    )
    
    # evaluate the model
    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=mmlu_reward_fn,
        prompts=prompts,
        answers=ground_truth,
        eval_sampling_params=sampling_params,
        output_filepath=f"../results/baseline_mmlu.jsonl"
    )
