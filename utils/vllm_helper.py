import os
import gc
import tqdm
import json
import torch
from tqdm import tqdm
from unittest.mock import patch
from typing import Callable, List

# Disable V1 multiprocessing so that model_executor is accessible
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.95):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch(
        "torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            enforce_eager=True,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def init_policy(model_id='Qwen/Qwen2.5-Math-1.5B', device=None):
    ''' Initialize the policy model and tokenizer '''
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.to(device)
    return model, tokenizer


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """ Load the weights from a HuggingFace model into a vLLM instance."""
    weights = list(policy.state_dict().items())
    llm.apply_model(lambda model: model.load_weights(weights))


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable,
    prompts: List[str],  # questions with prompt template
    answers: List[str],  # ground truth answers
    eval_sampling_params: SamplingParams,
    output_filepath: str = "",
) -> dict:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    Parameters
    ----------
    reward_fn : callable
        ``(responses: list[str], ground_truths: list) -> list[dict]``
        Batch reward function with a uniform interface for all dataset
        types (math, code, gsm8k, …).

    Returns a dict with keys: accuracy, format_accuracy, answer_accuracy.
    """
    # generate outputs for each example using the vLLM engine
    print(f"\nGenerating responses for {len(prompts)} problems...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # compute per-sample token statistics (prompt + response)
    prompt_lens = [len(out.prompt_token_ids) for out in outputs]
    response_lens = [len(out.outputs[0].token_ids) for out in outputs]
    total_lens = [p + r for p, r in zip(prompt_lens, response_lens)]
    max_idx = max(range(len(total_lens)), key=lambda i: total_lens[i])
    print(f"✅ Generation complete.")
    print(f"  Prompt  tokens — max: {max(prompt_lens)}, "
          f"min: {min(prompt_lens)}, avg: {sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"  Response tokens — max: {max(response_lens)}, "
          f"min: {min(response_lens)}, avg: {sum(response_lens)/len(response_lens):.0f}")
    print(f"  Total   tokens — max: {max(total_lens)} (sample #{max_idx}), "
          f"min: {min(total_lens)}, avg: {sum(total_lens)/len(total_lens):.0f}")

    del vllm_model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # prepare to compute metrics
    format_correct = 0
    answer_correct = 0
    total_correct = 0
    results = []

    generated_texts = [out.outputs[0].text.strip() for out in outputs]

    print("Evaluating responses and computing metrics...")
    reward_dicts = reward_fn(generated_texts, answers)
    for i, (output, reward_dict) in enumerate(zip(outputs, reward_dicts)):
        format_correct += int(reward_dict.get("format_reward", 0))
        answer_correct += int(reward_dict.get("answer_reward", 0))
        total_correct += int(reward_dict.get("reward", 0))
        results.append({
            "prompt": output.prompt,
            "response": generated_texts[i],
            "ground_truth": answers[i],
            "format_reward": reward_dict.get("format_reward", 0),
            "answer_reward": reward_dict.get("answer_reward", 0),
            "is_correct": reward_dict.get("reward", 0)
        })

    # calculate overall accuracy
    output_len = len(outputs)
    format_accuracy = (format_correct / output_len) * 100 if outputs else 0
    answer_accuracy = (answer_correct / output_len) * 100 if outputs else 0
    accuracy = (total_correct / output_len) * 100 if outputs else 0
    print(f"\n✅ Evaluation complete.")
    print(f"📈 Format Accuracy: {format_accuracy:.2f}% ({int(format_correct)}/{output_len})")
    print(f"📈 Answer Accuracy: {answer_accuracy:.2f}% ({int(answer_correct)}/{output_len})")
    print(f"📈 Overall Accuracy: {accuracy:.2f}% ({int(total_correct)}/{output_len})")

    # serialize the detailed results to a JSON file for later analysis
    if output_filepath is not None and output_filepath != "":
        print(f"\nSerializing results to {output_filepath}...")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w") as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")
        print(f"✅ Results saved successfully to {output_filepath}.")

    return {
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "answer_accuracy": answer_accuracy,
    }
