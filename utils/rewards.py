import re
from typing import List

from tqdm import tqdm

from .math_utils import extract_answer, grade
from .code_utils import evaluate_code_response


# ---------------------------------------------------------------------------
# Public reward functions — all share a uniform batch interface:
#
#   fn(responses: list[str], ground_truths: list) -> list[dict]
#
# For CPU-bound tasks (math, gsm8k, mmlu) the loop is sequential; for code
# tasks, evaluation is parallelised via ThreadPoolExecutor since each code
# evaluation spawns a subprocess whose I/O releases the GIL.
# ---------------------------------------------------------------------------

def mmlu_reward_fn(responses: List[str], ground_truths: List[str]) -> List[dict]:
    """Batch reward function for MMLU evaluation."""
    results = []
    for response, ground_truth in tqdm(zip(responses, ground_truths),
                                       total=len(responses), desc="MMLU Reward"):
        prediction = None
        if response is not None:
            # 1. Try the explicit sentence format first
            match = re.search(r"The correct answer is ([A-D])", response)
            if match:
                prediction = match.group(1)
            else:
                # 2. Try common variants: "answer is A", "Answer: A", etc.
                match = re.search(r"[Aa]nswer\s*(?:is|:)\s*([A-D])\b", response)
                if match:
                    prediction = match.group(1)
                else:
                    # 3. Bare letter: response is just "A"-"D" (possibly with
                    #    trailing content like "B. some text" from the option)
                    match = re.match(r"\s*([A-D])\b", response)
                    if match:
                        prediction = match.group(1)

        if prediction is not None and prediction == ground_truth:
            results.append({"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0})
        elif prediction is not None:
            results.append({"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0})
        else:
            results.append({"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0})
    return results


def gsmk_reward_fn(responses: List[str], ground_truths: List[str]) -> List[dict]:
    """Batch reward function that compares parsed model completions against ground truths."""
    def parse_gsm8k_response(response: str) -> float:
        '''Parses the language model output to extract the final numeric answer.'''
        if response is None: return None
        # Prefer the number after "####" if the model uses the GSM8K marker
        if "####" in response:
            after_marker = response.split("####")[-1].strip()
            pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
            m = re.search(pattern, after_marker)
            if m:
                try:
                    return float(m.group().replace(',', ''))
                except ValueError:
                    pass
        # Fallback: take the last number in the entire response
        pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
        matches = re.findall(pattern, response)
        if not matches: return None
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            return None

    results = []
    for completion, ground_truth in tqdm(zip(responses, ground_truths),
                                         total=len(responses), desc="GSM8K Reward"):
        # 1. parse the model's prediction
        pred_val = parse_gsm8k_response(completion)
        if pred_val is None:
            results.append({"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0})
            continue

        # 2. parse the ground truth answer
        if "####" in str(ground_truth):
            gt_str = str(ground_truth).split("####")[-1].strip()
        else:
            gt_str = str(ground_truth).strip()
        gt_val = parse_gsm8k_response(gt_str)
        if gt_val is None:
            results.append({"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0})
            continue

        # 3. compare and assign reward
        is_correct = 1.0 if abs(pred_val - gt_val) < 0.01 else 0.0
        results.append({
            "format_reward": 1.0,
            "answer_reward": is_correct,
            "reward": is_correct,
        })
    return results


def question_only_reward_fn(responses: List[str], ground_truths: List) -> List[dict]:
    """Batch reward function for question-only evaluation."""
    results = []
    for response, ground_truth in tqdm(zip(responses, ground_truths),
                                       total=len(responses), desc="QOnly Reward"):
        model_answer = extract_answer(response)
        if model_answer is None:
            results.append({"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0})
            continue

        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        if isinstance(ground_truth, str):
            is_correct = grade(model_answer, ground_truth, True)
        elif isinstance(ground_truth, list):
            is_correct = False
            for gt in ground_truth:
                is_correct |= grade(model_answer, gt, True)

        if is_correct:
            results.append({"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0})
        else:
            results.append({"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0})
    return results


def dsr1_reward_fn(responses: List[str], ground_truths: List) -> List[dict]:
    """Batch reward function for R1-zero like math evaluation.

    Evaluates model-generated responses by:
      1. Checking format — response must contain <think>...</think><answer>...</answer> tags.
      2. Extracting the answer from within <answer> tags (handles \\boxed{} notation).
      3. Grading the extracted answer against ground_truth (supports str or list of str).

    Reward structure:
      - format_reward:  1.0 if response follows the expected format, 0.0 otherwise.
      - answer_reward:  1.0 if the extracted answer matches ground_truth, 0.0 otherwise.
      - reward:         1.0 only when both formatted AND correct, 0.0 otherwise.
    """
    results = []
    for response, ground_truth in tqdm(zip(responses, ground_truths),
                                       total=len(responses), desc="DSR1 Reward"):
        if re.search(r"</think>.*?<answer>", response, re.DOTALL) and "</answer>" in response:
            model_answer = response.split("<answer>")[-1].replace("</answer>", "")

            if "\\boxed" in model_answer:
                model_answer = extract_answer(model_answer)
                if model_answer is None:
                    results.append({"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0})
                    continue

            if isinstance(ground_truth, float) or isinstance(ground_truth, int):
                ground_truth = str(ground_truth)

            if isinstance(ground_truth, str):
                is_correct = grade(model_answer, ground_truth, False)
            elif isinstance(ground_truth, list):
                is_correct = False
                for gt in ground_truth:
                    is_correct |= grade(model_answer, gt, False)

            if is_correct:
                results.append({"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0})
            else:
                results.append({"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0})
        else:
            results.append({"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0})
    return results


def code_reward_fn(responses, ground_truths, timeout=10.0,
                   test_type=None, max_workers=None):
    """Reward function for verifiable code generation tasks.

    Evaluates all code responses in parallel via a thread pool.
    Returns a list of reward dicts.
    """
    results = evaluate_code_response(
        responses=responses,
        test_cases_list=ground_truths,
        timeout=timeout,
        test_type=test_type,
        max_workers=max_workers,
    )
    return [
        {
            "format_reward":  r["format_reward"],
            "answer_reward":  r["answer_reward"],
            "partial_reward": r["partial_reward"],
            "reward":         r["reward"],
        }
        for r in results
    ]
