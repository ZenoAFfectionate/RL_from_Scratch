import re

from .math_utils import extract_answer, grade
from .code_utils import evaluate_code_response


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


def gsmk_reward_fn(completion: str, ground_truth: str) -> float:
    """Reward function that compares the parsed model completion against the ground truth."""
    def parse_gsm8k_response(response: str) -> float:
        '''Parses the language model output to extract the final numeric answer.'''
        if response is None: return None
        # create pattern and find all matches
        pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
        matches = re.findall(pattern, response)

        if not matches: return None
        # take final number and remove commas
        last_number_str = matches[-1]
        clean_number_str = last_number_str.replace(',', '')

        try:
            return float(clean_number_str)
        except ValueError:
            return None
    
    # 1. parse the model's prediction
    pred_val = parse_gsm8k_response(completion)
    if pred_val is None: return {"reward": 0.0, "prediction": None}

    if "####" in str(ground_truth):
        gt_str = str(ground_truth).split("####")[-1].strip()
    else:
        gt_str = str(ground_truth).strip()

    # 2. parse the ground truth answer
    gt_val = parse_gsm8k_response(gt_str)
    if gt_val is None: return {"reward": 0.0, "prediction": None}

    # 3. compare and assign reward
    reward = 1.0 if abs(pred_val - gt_val) < 0.01 else 0.0
    return {"reward": reward, "prediction": pred_val}


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


def dsr1_reward_fn(response, ground_truth, fast=False):
    """
    Reward function for R1-zero like math evaluation.

    Evaluates a model-generated response by:
      1. Checking format — response must contain <think>...</think><answer>...</answer> tags.
      2. Extracting the answer from within <answer> tags (handles \\boxed{} notation).
      3. Grading the extracted answer against ground_truth (supports str or list of str).

    Reward structure:
      - format_reward:  1.0 if response follows the expected format, 0.0 otherwise.
      - answer_reward:  1.0 if the extracted answer matches ground_truth, 0.0 otherwise.
      - reward:         1.0 only when both formatted AND correct, 0.0 otherwise.
    """
    if re.search(r"</think>\s*<answer>", response) and "</answer>" in response:
        model_answer = response.split("<answer>")[-1].replace("</answer>", "")
        #
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
        
        # check the correctness of model answer:
        if isinstance(ground_truth, str):
            is_correct = grade(model_answer, ground_truth, fast)
        elif isinstance(ground_truth, list):
            is_correct = False
            for gt in ground_truth:
                is_correct |= grade(model_answer, gt, fast)

        if is_correct:
            # formatted and right answer;
            return {
                "format_reward": 1.0,
                "answer_reward": 1.0,
                "reward": 1.0
            }
        else:
            # formatted but wrong answer;
            return {
                "format_reward": 1.0,
                "answer_reward": 0.0,
                "reward": 0.0
            }
    else:
        # both unformatted and wrong answer...
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }


def code_reward_fn(response, ground_truth, timeout=10.0, test_type=None):
    """
    Reward function for verifiable code generation tasks.

    Evaluates a model-generated code response by:
      1. Extracting code from the response (handles markdown, XML tags, raw).
      2. Checking syntax — a compilable solution earns format_reward = 1.0.
      3. Running extracted code against provided test cases —
         passing ALL tests earns answer_reward = 1.0.

    Reward structure (mirrors dsr1_reward_fn):
      - format_reward:  1.0 if code compiles,       0.0 otherwise.
      - answer_reward:  1.0 if all tests pass,      0.0 otherwise.
      - partial_reward: fraction of tests passed (for softer shaping).
      - reward:         1.0 when ALL tests pass, 0.0 otherwise.
    """
    result = evaluate_code_response(
        response=response,
        test_cases=ground_truth,
        timeout=timeout,
        test_type=test_type,
    )

    return {
        "format_reward":  result["format_reward"],
        "answer_reward":  result["answer_reward"],
        "partial_reward": result["partial_reward"],
        "reward":         result["reward"],
    }