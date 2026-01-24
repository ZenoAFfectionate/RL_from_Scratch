import os
import json
import argparse

from vllm import LLM, SamplingParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AlpacaEval predictions with vLLM.")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="The name or path of the model to use."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="alpa",
        help="The dataset to evaluate on."
    )

    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--min_tokens", type=int, default=2,   help="Minimum number of tokens to generate.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate.")

    args = parser.parse_args()

    print(f"Running AlpacaEval generation with parameters:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Temperature: {args.temperature} (Greedy)")
    print(f"  - Top-p: {args.top_p}")

    # load AlpacaEval instructions
    print(f"Loading validation set ...", end=' ')
    with open(f"../data/{args.dataset_name}/alpaca_eval.json", "r") as f:
        dataset = json.load(f)
    print(f"Success!\nLoaded {len(dataset)} examples.")

    # prepare instruction for vllm
    prompts = [example["instruction"] for example in dataset]

    # initialize model by means of vLLM
    print(f"Initializing vLLM with model: {args.model_name}")
    vllm_model = LLM(model=args.model_name, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        # do not want strict stop tokens
    )

    print("Generating outputs...")
    outputs = vllm_model.generate(prompts, sampling_params)

    # identifier for generator based on the model name
    generator_id = os.path.basename(args.model_name)

    results = []
    for i, request_output in enumerate(outputs):
        generated_text = request_output.outputs[0].text
        
        entry = {
            "instruction": dataset[i]["instruction"],
            "output": generated_text,
            "generator": generator_id,
            "dataset": dataset[i]["dataset"]
        }
        results.append(entry)

    # save results to file
    with open("../results/baseline_alpa.jsonl", "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)
        
    print("Done!")
