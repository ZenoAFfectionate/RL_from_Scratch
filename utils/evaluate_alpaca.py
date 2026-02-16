"""
AlpacaEval evaluation using a local vLLM judge model.

Replaces the alpaca_eval package with a standalone vLLM-based evaluator.
Loads a local judge model (e.g., Qwen3-72B-Instruct with FP8 quantization)
to compare candidate outputs against reference outputs via pairwise ranking.

Usage:
    python utils/evaluate_alpaca.py \
        --model-outputs ../results/baseline_alpa.json \
        --judge-model Qwen/Qwen3-72B-Instruct \
        --quantization fp8 \
        --num-gpus 2
"""
import ast
import argparse
import json
import os
import random
import re
import sys

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ── Judge prompt (adapted from AlpacaEval) ────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge that evaluates the quality of AI assistant "
    "responses. You will be given an instruction and two responses from "
    "different models. Your job is to rank them based on overall quality."
)

JUDGE_USER_TEMPLATE = """\
I want you to compare two AI assistant responses to the same instruction. \
Please rank them based on which response would be preferred by humans, \
considering the following criteria:
- Helpfulness: Does it address the user's request?
- Accuracy: Is the information correct?
- Relevance: Does it stay on topic without unnecessary content?
- Depth: Does it provide sufficient detail?
- Clarity: Is it well-organized and easy to understand?

Here is the instruction:
{{
    "instruction": \"\"\"{instruction}\"\"\"
}}

Here are the outputs of the two models:
[
    {{
        "model": "model_1",
        "answer": \"\"\"{output_1}\"\"\"
    }},
    {{
        "model": "model_2",
        "answer": \"\"\"{output_2}\"\"\"
    }}
]

Now please rank the models by the quality of their answers, so that the \
model with rank 1 has the best output. Then return a list of the model \
names and ranks, i.e., produce the following output:
[
    {{'model': <model-name>, 'rank': <model-rank>}},
    {{'model': <model-name>, 'rank': <model-rank>}}
]

Your response must be a valid Python list of dictionaries and should \
contain nothing else because we will directly execute it in Python. \
Please provide the ranking that the majority of humans would give."""


# ── Ranking parser ────────────────────────────────────────────────────────

def parse_ranking(response_text: str) -> str:
    """Parse the judge response and return 'model_1', 'model_2', or 'tie'."""
    text = response_text.strip()

    # 1. Try ast.literal_eval on the whole response
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list) and len(result) == 2:
            ranks = {item["model"]: item["rank"] for item in result}
            r1, r2 = ranks.get("model_1", 2), ranks.get("model_2", 2)
            if r1 < r2:
                return "model_1"
            elif r2 < r1:
                return "model_2"
            return "tie"
    except Exception:
        pass

    # 2. Try ast.literal_eval on a bracketed substring (model may add text around it)
    bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
    if bracket_match:
        try:
            result = ast.literal_eval(bracket_match.group())
            if isinstance(result, list) and len(result) == 2:
                ranks = {item["model"]: item["rank"] for item in result}
                r1, r2 = ranks.get("model_1", 2), ranks.get("model_2", 2)
                if r1 < r2:
                    return "model_1"
                elif r2 < r1:
                    return "model_2"
                return "tie"
        except Exception:
            pass

    # 3. Regex fallback: find rank assignments
    rank1 = re.search(r"['\"]model_1['\"].*?['\"]?rank['\"]?\s*:\s*(\d)", text)
    rank2 = re.search(r"['\"]model_2['\"].*?['\"]?rank['\"]?\s*:\s*(\d)", text)
    if rank1 and rank2:
        r1, r2 = int(rank1.group(1)), int(rank2.group(1))
        if r1 < r2:
            return "model_1"
        elif r2 < r1:
            return "model_2"
        return "tie"

    return "tie"


# ── Main ──────────────────────────────────────────────────────────────────

def evaluate(
    model_outputs_path: str,
    reference_outputs_path: str,
    judge_model: str,
    quantization: str | None = "fp8",
    num_gpus: int = 2,
    max_model_len: int = 8192,
    output_path: str | None = None,
    seed: int = 42,
):
    random.seed(seed)

    # ── Load data ─────────────────────────────────────────────────────
    print(f"Loading candidate outputs from {model_outputs_path}")
    with open(model_outputs_path) as f:
        candidates = json.load(f) if model_outputs_path.endswith(".json") \
            else [json.loads(l) for l in f]

    print(f"Loading reference outputs from {reference_outputs_path}")
    with open(reference_outputs_path) as f:
        references = json.load(f)

    assert len(candidates) == len(references), \
        f"Mismatch: {len(candidates)} candidates vs {len(references)} references"

    # ── Build judge prompts ───────────────────────────────────────────
    print(f"Loading tokenizer: {judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)

    judge_prompts = []
    order_map = []  # track position: "candidate_first" or "reference_first"

    for cand, ref in zip(candidates, references):
        instruction = cand["instruction"]
        cand_output = cand["output"]
        ref_output = ref["output"]

        # Randomize order to reduce position bias
        if random.random() < 0.5:
            o1, o2 = cand_output, ref_output
            order_map.append("candidate_first")
        else:
            o1, o2 = ref_output, cand_output
            order_map.append("reference_first")

        user_content = JUDGE_USER_TEMPLATE.format(
            instruction=instruction, output_1=o1, output_2=o2,
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Disable thinking for judge (Qwen3-specific; harmless for others)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        judge_prompts.append(prompt)

    # ── Initialize judge model ────────────────────────────────────────
    print(f"Initializing judge: {judge_model}  (quantization={quantization}, tp={num_gpus})")
    model = LLM(
        model=judge_model,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=max_model_len,
        quantization=quantization,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=256,
    )

    # ── Run judge ─────────────────────────────────────────────────────
    print(f"Judging {len(judge_prompts)} comparisons ...")
    raw_outputs = model.generate(judge_prompts, sampling_params)

    # ── Parse results ─────────────────────────────────────────────────
    wins, losses, ties = 0, 0, 0
    details = []

    for i, output in enumerate(tqdm(raw_outputs, desc="Parsing rankings")):
        response = output.outputs[0].text.strip()
        winner = parse_ranking(response)

        # Map back: "model_1" always refers to the first output shown
        if order_map[i] == "candidate_first":
            result = {"model_1": "win", "model_2": "loss"}.get(winner, "tie")
        else:
            result = {"model_2": "win", "model_1": "loss"}.get(winner, "tie")

        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            ties += 1

        details.append({
            "instruction": candidates[i]["instruction"],
            "candidate_output": candidates[i]["output"],
            "reference_output": references[i]["output"],
            "judge_response": response,
            "order": order_map[i],
            "result": result,
        })

    # ── Report ────────────────────────────────────────────────────────
    total = len(raw_outputs)
    win_rate = (wins + ties * 0.5) / total * 100

    cand_gen = candidates[0].get("generator", "unknown")
    ref_gen = references[0].get("generator", "gpt4_1106_preview")

    report = (
        f"\n{'='*60}\n"
        f"  AlpacaEval Results\n"
        f"{'='*60}\n"
        f"  Candidate : {cand_gen}\n"
        f"  Reference : {ref_gen}\n"
        f"  Judge     : {judge_model}\n"
        f"{'='*60}\n"
        f"  Wins      : {wins:>4}/{total}  ({wins/total*100:5.1f}%)\n"
        f"  Losses    : {losses:>4}/{total}  ({losses/total*100:5.1f}%)\n"
        f"  Ties      : {ties:>4}/{total}  ({ties/total*100:5.1f}%)\n"
        f"  Win Rate  : {win_rate:.2f}%\n"
        f"{'='*60}\n"
    )
    print(report)

    # ── Save detailed results ─────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": {
                    "win_rate": win_rate,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "total": total,
                    "candidate_generator": cand_gen,
                    "reference_generator": ref_gen,
                    "judge_model": judge_model,
                },
                "details": details,
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to {output_path}")

    return win_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlpacaEval pairwise evaluation using a local vLLM judge.")
    parser.add_argument("--model-outputs", type=str, required=True,
                        help="Path to candidate model outputs (JSON or JSONL)")
    parser.add_argument("--reference-outputs", type=str, default=None,
                        help="Path to reference outputs (JSON). "
                             "Default: ../data/alpa/alpaca_eval_gpt4_baseline.json")
    parser.add_argument("--judge-model", type=str,
                        default="Qwen/Qwen3-72B-Instruct",
                        help="Judge model name or local path")
    parser.add_argument("--quantization", type=str, default="fp8",
                        help="Quantization: fp8, awq, gptq, or none")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="Number of GPUs (tensor parallel)")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save detailed JSON results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Default reference path
    if args.reference_outputs is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.reference_outputs = os.path.join(
            project_root, "data", "alpa", "alpaca_eval_gpt4_baseline.json")

    quant = None if args.quantization.lower() == "none" else args.quantization

    evaluate(
        model_outputs_path=args.model_outputs,
        reference_outputs_path=args.reference_outputs,
        judge_model=args.judge_model,
        quantization=quant,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        output_path=args.output_path,
        seed=args.seed,
    )
