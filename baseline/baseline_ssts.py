import os
import json
import argparse

from vllm import LLM, SamplingParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SimpleSafetyTests predictions with vLLM.")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_name", type=str, default="ssts")

    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate.")

    args = parser.parse_args()

    print(f"Running AlpacaEval generation with parameters:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Temperature: {args.temperature} (Greedy)")
    print(f"  - Top-p: {args.top_p}")

    # load AlpacaEval instructions
    dataset = []
    print(f"Loading validation set ...", end=' ')
    with open(f"../data/{args.dataset_name}/sst_eval.jsonl", "r") as f:
        for line in f: dataset.append(json.loads(line))
    print(f"Success!\nLoaded {len(dataset)} examples.")

    # prepare instruction for vllm
    prompts = [example["prompt"] for example in dataset]

    # initialize model by means of vLLM
    print(f"Initializing vLLM with model: {args.model_name}")
    vllm_model = LLM(model=args.model_name, trust_remote_code=True, max_model_len=4096)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print("Generating outputs...")
    outputs = vllm_model.generate(prompts, sampling_params)

    # identifier for generator based on the model name
    generator_id = os.path.basename(args.model_name)

    results = []
    for i, request_output in enumerate(outputs):
        generated_text = request_output.outputs[0].text
        original_instruction = dataset[i].get("prompt", "")
        
        entry = {
            "prompts_final": original_instruction,
            "output": generated_text,
            "generator": generator_id,
            "id": dataset[i].get("id", i),
            "harm_area": dataset[i].get("harm_area", "unknown")
        }
        results.append(entry)

    # save results to file (JSONL format: one JSON object per line)
    with open("../results/baseline_ssts.jsonl", "w", encoding="utf-8") as fout:
        for entry in results: fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
    print(f"Saved {len(results)} results to ../results/baseline_ssts.jsonl")
    print("Done!")
