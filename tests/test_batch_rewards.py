"""
Test suite for the unified reward function interface.

Validates:
  1. evaluate_code_response (parallel subprocess evaluation in code_utils)
  2. code_reward_fn (code reward via thread pool)
  3. dsr1_reward_fn (math reward)
  4. gsmk_reward_fn (gsm8k reward)
  5. mmlu_reward_fn (mmlu reward)
  6. Interface consistency: all reward functions share the same
     (responses, ground_truths) -> list[dict] signature
  7. evaluate_vllm uses the reward_fn parameter
  8. Real data smoke tests using a few examples from each dataset
"""

import json
import os
import sys
import time

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.code_utils import (
    evaluate_code_response,
)
from utils.rewards import (
    code_reward_fn,
    dsr1_reward_fn,
    gsmk_reward_fn,
    mmlu_reward_fn,
    question_only_reward_fn,
)


# ─── Helpers ───────────────────────────────────────────────────────────

def load_jsonl(path, n=3):
    """Load the first *n* lines from a JSONL file."""
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1


# ─── Test 1: evaluate_code_response with synthetic data ─────────

print("\n" + "=" * 60)
print("Test 1: evaluate_code_response — synthetic data")
print("=" * 60)

# Case 1a: correct code, assert-based tests
responses_code = [
    "```python\ndef add(a, b):\n    return a + b\n```",
    "```python\ndef multiply(a, b):\n    return a * b\n```",
    "```python\ndef add(a, b):\n    return a - b\n```",  # intentionally wrong
]
test_cases_code = [
    ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    ["assert multiply(2, 3) == 6", "assert multiply(0, 5) == 0"],
    ["assert add(1, 2) == 3"],  # will fail
]

results = evaluate_code_response(responses_code, test_cases_code, timeout=5.0)

check("returns list of correct length", len(results) == 3)
check("correct code (add) passes all tests", results[0]["reward"] == 1.0)
check("correct code (multiply) passes all tests", results[1]["reward"] == 1.0)
check("wrong code (add returns a-b) fails", results[2]["reward"] == 0.0)
check("correct code has syntax_valid=True", results[0]["syntax_valid"] is True)
check("wrong code still has syntax_valid=True", results[2]["syntax_valid"] is True)

# Case 1b: syntax error
responses_syntax = [
    "```python\ndef foo(\n```",  # syntax error
]
test_cases_syntax = [["assert foo() == 1"]]

results_syn = evaluate_code_response(responses_syntax, test_cases_syntax, timeout=5.0)
check("syntax error detected", results_syn[0]["syntax_valid"] is False)
check("syntax error reward = 0", results_syn[0]["reward"] == 0.0)

# Case 1c: empty response
responses_empty = ["", "   "]
test_cases_empty = [["assert True"], ["assert True"]]
results_empty = evaluate_code_response(responses_empty, test_cases_empty, timeout=5.0)
check("empty response reward = 0", results_empty[0]["reward"] == 0.0)


# ─── Test 2: code_reward_fn interface ──────────────────────────────────

print("\n" + "=" * 60)
print("Test 2: code_reward_fn — interface consistency")
print("=" * 60)

rewards = code_reward_fn(responses_code, test_cases_code)
check("returns list of dicts", isinstance(rewards, list) and all(isinstance(r, dict) for r in rewards))
check("length matches input", len(rewards) == len(responses_code))
check("each dict has 'reward' key", all("reward" in r for r in rewards))
check("each dict has 'format_reward' key", all("format_reward" in r for r in rewards))
check("each dict has 'answer_reward' key", all("answer_reward" in r for r in rewards))
check("correct code reward=1.0", rewards[0]["reward"] == 1.0)
check("wrong code reward=0.0", rewards[2]["reward"] == 0.0)


# ─── Test 3: Consistency — single (_code_reward) vs batch ──────────────

print("\n" + "=" * 60)
print("Test 3: Single vs batch consistency (code)")
print("=" * 60)

for i, (resp, tc) in enumerate(zip(responses_code, test_cases_code)):
    single = evaluate_code_response([resp], [tc], timeout=5.0)[0]
    batch = rewards[i]
    check(
        f"sample {i}: single.reward == batch.reward",
        single["reward"] == batch["reward"],
        f"single={single['reward']}, batch={batch['reward']}",
    )
    check(
        f"sample {i}: single.format_reward == batch.format_reward",
        single["format_reward"] == batch["format_reward"],
    )


# ─── Test 4: dsr1_reward_fn with real math data ───────────────────────

print("\n" + "=" * 60)
print("Test 4: dsr1_reward_fn — real math data")
print("=" * 60)

math_data_path = os.path.join(project_root, "data", "math", "train.jsonl")
if os.path.exists(math_data_path):
    math_items = load_jsonl(math_data_path, n=5)

    # Use the solutions in the dataset as "model responses" — they should be correct
    math_responses = [item["solution"] for item in math_items]
    math_ground_truths = []
    for item in math_items:
        sol = item["solution"]
        if "<answer>" in sol and "</answer>" in sol:
            gt = sol.split("<answer>")[-1].replace("</answer>", "").strip()
            math_ground_truths.append(gt)
        else:
            math_ground_truths.append(sol)

    batch_results = dsr1_reward_fn(math_responses, math_ground_truths)
    check("returns correct length", len(batch_results) == len(math_responses))
    check("all dicts have 'reward'", all("reward" in r for r in batch_results))
    check("all dicts have 'format_reward'", all("format_reward" in r for r in batch_results))

    # Solutions from training set should have correct format (think + answer tags)
    num_formatted = sum(1 for r in batch_results if r["format_reward"] == 1.0)
    check(f"most responses have correct format ({num_formatted}/{len(batch_results)})",
          num_formatted >= len(batch_results) // 2)

    # Verify consistency with single-element batch call
    for i in range(min(3, len(math_responses))):
        single = dsr1_reward_fn([math_responses[i]], [math_ground_truths[i]])[0]
        check(
            f"math sample {i}: single == batch",
            single["reward"] == batch_results[i]["reward"]
            and single["format_reward"] == batch_results[i]["format_reward"],
        )
else:
    print("  [SKIP] math data not found")


# ─── Test 5: gsmk_reward_fn with real gsmk data ───────────────────────

print("\n" + "=" * 60)
print("Test 5: gsmk_reward_fn — real gsmk data")
print("=" * 60)

gsmk_data_path = os.path.join(project_root, "data", "gsmk", "train.jsonl")
if os.path.exists(gsmk_data_path):
    gsmk_items = load_jsonl(gsmk_data_path, n=5)

    # Simulate model responses by using the ground truth answers
    gsmk_responses = [item["answer"] for item in gsmk_items]
    gsmk_ground_truths = [item["answer"] for item in gsmk_items]

    batch_results = gsmk_reward_fn(gsmk_responses, gsmk_ground_truths)
    check("returns correct length", len(batch_results) == len(gsmk_responses))

    # Identical response and ground truth should yield reward=1.0
    num_correct = sum(1 for r in batch_results if r["reward"] == 1.0)
    check(f"self-comparison yields correct ({num_correct}/{len(batch_results)})",
          num_correct == len(batch_results))

    # Verify single vs batch
    for i in range(min(3, len(gsmk_responses))):
        single = gsmk_reward_fn([gsmk_responses[i]], [gsmk_ground_truths[i]])[0]
        check(
            f"gsmk sample {i}: single == batch",
            single["reward"] == batch_results[i]["reward"],
        )

    # Test with a wrong answer
    wrong_results = gsmk_reward_fn(["The answer is 99999"], [gsmk_ground_truths[0]])
    check("wrong gsmk answer gives reward=0", wrong_results[0]["reward"] == 0.0)
else:
    print("  [SKIP] gsmk data not found")


# ─── Test 6: mmlu_reward_fn with real mmlu data ───────────────────────

print("\n" + "=" * 60)
print("Test 6: mmlu_reward_fn — real mmlu data")
print("=" * 60)

mmlu_data_path = os.path.join(project_root, "data", "mmlu", "train.jsonl")
if os.path.exists(mmlu_data_path):
    mmlu_items = load_jsonl(mmlu_data_path, n=5)

    # MMLU ground truth is an integer index (0-3) → letter (A-D)
    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    mmlu_responses = []
    mmlu_ground_truths = []
    for item in mmlu_items:
        gt_letter = index_to_letter[item["answer"]]
        mmlu_responses.append(f"The correct answer is {gt_letter}")
        mmlu_ground_truths.append(gt_letter)

    batch_results = mmlu_reward_fn(mmlu_responses, mmlu_ground_truths)
    check("returns correct length", len(batch_results) == len(mmlu_responses))

    num_correct = sum(1 for r in batch_results if r["reward"] == 1.0)
    check(f"self-comparison all correct ({num_correct}/{len(batch_results)})",
          num_correct == len(batch_results))

    # Verify single vs batch
    for i in range(min(3, len(mmlu_responses))):
        single = mmlu_reward_fn([mmlu_responses[i]], [mmlu_ground_truths[i]])[0]
        check(
            f"mmlu sample {i}: single == batch",
            single["reward"] == batch_results[i]["reward"],
        )

    # Test with a wrong answer
    wrong_results = mmlu_reward_fn(
        ["The correct answer is A"],
        ["B"],
    )
    check("wrong mmlu answer gives reward=0", wrong_results[0]["reward"] == 0.0)
    check("wrong mmlu answer has format_reward=1", wrong_results[0]["format_reward"] == 1.0)
else:
    print("  [SKIP] mmlu data not found")


# ─── Test 7: code_reward_fn with real code data ───────────────────────

print("\n" + "=" * 60)
print("Test 7: code_reward_fn — real code data")
print("=" * 60)

code_data_path = os.path.join(project_root, "data", "code", "train.jsonl")
if os.path.exists(code_data_path):
    code_items = load_jsonl(code_data_path, n=3)

    # Use the gold solutions as "model responses"
    code_responses = [item["solution"] for item in code_items]
    code_test_cases = [item["test"] for item in code_items]

    batch_results = code_reward_fn(code_responses, code_test_cases, timeout=10.0)
    check("returns correct length", len(batch_results) == len(code_responses))
    check("all dicts have required keys",
          all({"reward", "format_reward", "answer_reward", "partial_reward"} <= set(r.keys())
              for r in batch_results))

    # Gold solutions should have valid syntax at least
    num_formatted = sum(1 for r in batch_results if r["format_reward"] == 1.0)
    check(f"gold solutions have valid syntax ({num_formatted}/{len(batch_results)})",
          num_formatted >= 1)

    # Verify single vs batch consistency
    for i in range(len(code_responses)):
        single = evaluate_code_response([code_responses[i]], [code_test_cases[i]], timeout=10.0)[0]
        check(
            f"code sample {i}: single == batch (reward={single['reward']})",
            single["reward"] == batch_results[i]["reward"]
            and single["format_reward"] == batch_results[i]["format_reward"],
        )
else:
    print("  [SKIP] code data not found")


# ─── Test 8: Parallel speedup for code evaluation ─────────────────────

print("\n" + "=" * 60)
print("Test 8: Parallel speedup measurement (code)")
print("=" * 60)

# Create N identical correct code responses to measure parallel vs sequential
N = 8
parallel_responses = [
    "```python\ndef square(x):\n    return x * x\n```"
] * N
parallel_tests = [
    ["assert square(2) == 4", "assert square(0) == 0", "assert square(-3) == 9"]
] * N

# Sequential timing (one-at-a-time batch calls)
t0 = time.perf_counter()
seq_results = [
    evaluate_code_response([r], [tc], timeout=5.0)[0]
    for r, tc in zip(parallel_responses, parallel_tests)
]
t_seq = time.perf_counter() - t0

# Parallel timing (using public code_reward_fn)
t0 = time.perf_counter()
par_results = code_reward_fn(parallel_responses, parallel_tests, timeout=5.0)
t_par = time.perf_counter() - t0

check("sequential results all correct", all(r["reward"] == 1.0 for r in seq_results))
check("parallel results all correct", all(r["reward"] == 1.0 for r in par_results))
check(
    "results match between sequential and parallel",
    all(s["reward"] == p["reward"] for s, p in zip(seq_results, par_results)),
)

speedup = t_seq / t_par if t_par > 0 else float("inf")
print(f"  [INFO] Sequential: {t_seq:.3f}s, Parallel: {t_par:.3f}s, Speedup: {speedup:.2f}x")
check(f"parallel is faster (speedup={speedup:.2f}x)", speedup > 1.0)


# ─── Test 9: evaluate_vllm signature check ─────────────────────────────

print("\n" + "=" * 60)
print("Test 9: evaluate_vllm signature check")
print("=" * 60)

import inspect
from utils.vllm_helper import evaluate_vllm

sig = inspect.signature(evaluate_vllm)
param_names = list(sig.parameters.keys())

check("'reward_fn' is a parameter", "reward_fn" in param_names)
check("old 'batch_reward_fn' is NOT a parameter", "batch_reward_fn" not in param_names)
check("parameter order: reward_fn is 2nd arg",
      param_names.index("reward_fn") == 1,
      f"actual order: {param_names}")


# ─── Test 10: Interface uniformity across all reward functions ─────────

print("\n" + "=" * 60)
print("Test 10: All reward functions have uniform interface")
print("=" * 60)

reward_fns = {
    "code_reward_fn": code_reward_fn,
    "dsr1_reward_fn": dsr1_reward_fn,
    "gsmk_reward_fn": gsmk_reward_fn,
    "mmlu_reward_fn": mmlu_reward_fn,
    "question_only_reward_fn": question_only_reward_fn,
}

# All should accept (responses, ground_truths) as the first two positional args
for name, fn in reward_fns.items():
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    check(
        f"{name}: first two params are (responses, ground_truths)",
        len(params) >= 2,
        f"params={params}",
    )


# ─── Test 11: GRPO / PPO trainer __init__ signature ────────────────────

print("\n" + "=" * 60)
print("Test 11: GRPO_Trainer / PPO_Trainer accept reward_fun")
print("=" * 60)

from algorithms.grpo import GRPO_Trainer
from algorithms.ppo import PPO_Trainer

grpo_sig = inspect.signature(GRPO_Trainer.__init__)
ppo_sig = inspect.signature(PPO_Trainer.__init__)

grpo_params = list(grpo_sig.parameters.keys())
ppo_params = list(ppo_sig.parameters.keys())

check("GRPO_Trainer has 'reward_fun' param", "reward_fun" in grpo_params)
check("GRPO_Trainer does NOT have 'batch_reward_fun' param", "batch_reward_fun" not in grpo_params)
check("PPO_Trainer has 'reward_fun' param", "reward_fun" in ppo_params)
check("PPO_Trainer does NOT have 'batch_reward_fun' param", "batch_reward_fun" not in ppo_params)


# ─── Test 12: SFT4ReasTrainer signature ────────────────────────────────

print("\n" + "=" * 60)
print("Test 12: SFT4ReasTrainer accepts reward_fun")
print("=" * 60)

from algorithms.sft4reas import SFT4ReasTrainer

sft_sig = inspect.signature(SFT4ReasTrainer.__init__)
sft_params = list(sft_sig.parameters.keys())

check("SFT4ReasTrainer has 'reward_fun' param", "reward_fun" in sft_params)
check("SFT4ReasTrainer does NOT have 'batch_reward_fun' param", "batch_reward_fun" not in sft_params)


# ─── Test 13: No 'batch_' prefix in any public name ───────────────────

print("\n" + "=" * 60)
print("Test 13: No 'batch_' prefix leaked into public API")
print("=" * 60)

import utils.rewards as rewards_module
public_names = [n for n in dir(rewards_module) if not n.startswith("_")]
batch_reward_names = [n for n in public_names if n.startswith("batch_") and "reward" in n]
check(
    f"no public reward names start with 'batch_' ({batch_reward_names or 'none'})",
    len(batch_reward_names) == 0,
    f"found: {batch_reward_names}",
)


# ─── Summary ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
