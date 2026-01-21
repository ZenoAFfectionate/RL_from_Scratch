import re

from .math_utils import extract_answer, grade


def mmlu_reward_fn(response: str, ground_truth: str) -> dict:
    """Reward function for MMLU evaluation."""
    prediction = None
    # extract the answer from the model response
    if response is not None:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match: prediction = match.group(1)
    # compare with ground truth and assign reward
    if prediction is not None and prediction == ground_truth:
        return {"reward": 1.0, "prediction": prediction}
    else:
        return {"reward": 0.0, "prediction": prediction}


def question_only_reward_fn(response, ground_truth, fast=True):
    """Reward function for question-only evaluation."""
    model_answer = extract_answer(response)
    if model_answer is None:
        # Cannot even parse anything.
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }
    if isinstance(ground_truth, float) or isinstance(ground_truth, int):
        ground_truth = str(ground_truth)
    if isinstance(ground_truth, str):
        is_correct = grade(model_answer, ground_truth, fast)
    elif isinstance(ground_truth, list):
        is_correct = False
        for gt in ground_truth:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        # Correctness reward.
        return {
            "format_reward": 1.0,
            "answer_reward": 1.0,
            "reward": 1.0
        }
    else:
        # Formatted but wrong answer; no format reward to avoid hacking.
        return {
            "format_reward": 1.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }


def r1_zero_reward_fn(response, ground_truth, fast=True):
    """Reward function for R1-zero like evaluation."""
    # We are strict about format to evaluate our models.
    if re.search(r"</think>\s*<answer>", response) and "</answer>" in response:
        model_answer = response.split("<answer>")[-1].replace("</answer>", "")
        if "\\boxed" in model_answer:
            model_answer = extract_answer(model_answer)
            if model_answer is None:
                return {
                    "format_reward": 1.0,
                    "answer_reward": 0.0,
                    "reward": 0.0
                }
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        if isinstance(ground_truth, str):
            is_correct = grade(model_answer, ground_truth, fast)
        elif isinstance(ground_truth, list):
            is_correct = False
            for gt in ground_truth:
                is_correct |= grade(model_answer, gt, fast)
        if is_correct:
            return {
                "format_reward": 1.0,
                "answer_reward": 1.0,
                "reward": 1.0
            }
        else:
            # Formatted but wrong answer; no format reward to avoid hacking.
            return {
                "format_reward": 1.0,
                "answer_reward": 0.0,
                "reward": 0.0
            }
    else:
        # Unformatted.
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }