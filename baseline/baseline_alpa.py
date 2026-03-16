import os
import re
import json
import argparse
import subprocess
import sys

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AlpacaEval predictions with vLLM.")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset_name", type=str, default="alpa")

    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum number of tokens to generate.")

    # Evaluation arguments
    parser.add_argument("--evaluate", action="store_true",
                        help="Run AlpacaEval judge after generation.")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-72B-Instruct",
                        help="Judge model for AlpacaEval evaluation.")
    parser.add_argument("--judge_quantization", type=str, default="fp8",
                        help="Judge quantization: fp8, awq, gptq, or none.")
    parser.add_argument("--judge_num_gpus", type=int, default=2,
                        help="Number of GPUs for the judge model.")

    args = parser.parse_args()

    print(f"Running AlpacaEval generation with parameters:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-p: {args.top_p}")

    # load AlpacaEval instructions
    print(f"Loading validation set ...", end=' ')
    with open(os.path.join(project_root, "data", args.dataset_name, "alpaca_eval.json"), "r") as f:
        dataset = json.load(f)
    print(f"Success!\nLoaded {len(dataset)} examples.")

    # Apply chat template so the model treats inputs as instructions
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts = []
    for example in dataset:
        messages = [{"role": "user", "content": example["instruction"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # initialize model by means of vLLM
    print(f"Initializing vLLM with model: {args.model_name}")
    vllm_model = LLM(
        model=args.model_name,
        trust_remote_code=True,
        max_model_len=32768,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print("Generating outputs...")
    outputs = vllm_model.generate(prompts, sampling_params)

    # identifier for generator based on the model name
    generator_id = os.path.basename(args.model_name)

    # Strip thinking tags from outputs
    _THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
    _THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)

    results = []
    for i, request_output in enumerate(outputs):
        generated_text = request_output.outputs[0].text
        # Remove closed <think>...</think>, then unclosed <think>... (truncated)
        generated_text = _THINK_CLOSED_RE.sub("", generated_text)
        generated_text = _THINK_UNCLOSED_RE.sub("", generated_text).strip()

        entry = {
            "instruction": dataset[i]["instruction"],
            "output": generated_text,
            "generator": generator_id,
            "dataset": dataset[i]["dataset"]
        }
        results.append(entry)

    # save results to file (JSON format for AlpacaEval compatibility)
    output_path = os.path.join(project_root, "results", "baseline_alpa.json")
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {output_path}")

    # Optional: run AlpacaEval judge
    if args.evaluate:
        # Free GPU memory from the generation model before loading the judge
        del vllm_model, outputs, tokenizer
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

        eval_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "utils", "evaluate_alpaca.py")
        eval_output = output_path.replace(".json", "_eval.json")

        cmd = [
            sys.executable, eval_script,
            "--model-outputs", output_path,
            "--judge-model", args.judge_model,
            "--quantization", args.judge_quantization,
            "--num-gpus", str(args.judge_num_gpus),
            "--output-path", eval_output,
        ]
        print(f"\nLaunching AlpacaEval judge: {args.judge_model}")
        subprocess.run(cmd, check=True)
    else:
        print("\nTo evaluate with a local judge, re-run with --evaluate, or:")
        print(f"  python utils/evaluate_alpaca.py "
              f"--model-outputs {output_path} "
              f"--judge-model Qwen/Qwen3-72B-Instruct "
              f"--quantization fp8 --num-gpus 2")

    print("Done!")
